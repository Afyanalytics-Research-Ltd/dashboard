"""Warehouse model tests: synthetic data in SQLite, no Snowflake.

Run:  cd analytics_app/dashboards/internal_reporting && python -m unittest discover -s patient_discovery/tests -t .
Critical-rule mutations:  python -m patient_discovery.tests.mutation_gate   (every mutation must fail this suite)
Set MODEL_REPORT=1 to print the replay and cross-check tables.
"""
import os
import re
import unittest

from patient_discovery.classify import family
from patient_discovery.model import measure as M
from patient_discovery.model.concepts import Statement
from patient_discovery.model.planner import understand

from . import model_world as W

REPORT = os.environ.get("MODEL_REPORT") == "1"
_CACHE = {}     # world -> observations: measurements are deterministic per world, so runs may share them


def run(world, wh, statements, **kw):
    r = understand(wh, statements, now=W.NOW, observations=_CACHE.get(world, ()), **kw)
    _CACHE[world] = r.observations
    return r


def text(a):
    parts = [q.text for q in a.questions] + a.grain.reasons + a.reasons + a.notes
    parts += [f"{c.column} {' '.join(c.reasons)}" for c in a.candidates]
    return " | ".join(parts)


def check_expected(test, A, expected):
    for f, (stage, must, must_not, fragments) in expected.items():
        a = A[f]
        with test.subTest(object=f):
            test.assertEqual(a.stage, stage, f"{a.reasons} {[q.code for q in a.questions]}")
            test.assertTrue(must <= a.codes(), f"missing {must - a.codes()}: {[q.text for q in a.questions]}")
            test.assertFalse(must_not & a.codes(), f"unexpected {must_not & a.codes()}")
            for frag in fragments:
                test.assertIn(frag, text(a))


class Scenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        objs, cls.st = W.scenario_world()
        cls.objs, cls.wh = objs, W.Warehouse(objs)
        cls.r = run("scenarios", cls.wh, cls.st)
        cls.A = cls.r.beliefs.assessments

    def test_expected_stages_codes_and_reasons(self):
        check_expected(self, self.A, W.EXPECTED_SCENARIOS)

    def test_expected_meanings_and_provenance(self):
        for f, (col, meaning, ns, frag) in W.EXPECTED_MEANING.items():
            c = next(c for c in self.A[f].candidates if c.column == col)
            with self.subTest(object=f):
                self.assertEqual((c.meaning, c.namespace), (meaning, ns), c.provenance)
                if frag:
                    self.assertIn(frag, " / ".join(c.provenance))

    def test_every_meaning_starts_at_a_statement_and_moves_only_by_derivation(self):
        ids = {s.id for s in self.st if s.kind == "namespace"}
        for a in self.A.values():
            for c in a.candidates:
                if c.meaning is None:
                    continue
                with self.subTest(object=a.fqn, column=c.column):
                    self.assertIn(c.provenance[0].split(":")[0] + ":" + c.provenance[0].split(":")[1], ids)
                    for step in c.provenance[1:]:
                        self.assertTrue(step.startswith(("aligned with", "lineage:")), step)
                    scope = next(s for s in self.st if s.id == c.namespace).assertion["scope"]["facility"]
                    self.assertEqual(a.facility, scope)

    def test_acceptance_requires_person_meaning(self):
        for a in self.A.values():
            if a.stage in ("auto", "replacement"):
                with self.subTest(object=a.fqn):
                    self.assertEqual(a.selected.meaning, "person")
                    self.assertTrue(a.selected.provenance)

    def test_household_register_regression(self):
        """Was a false acceptance: household numbers confirmed by a household 'register' (containment)."""
        a = self.A["HOSPITALS.LAKEVIEW_CLEAN.VISITS"]
        self.assertNotEqual(a.stage, "auto")
        self.assertIsNone(a.selected.meaning)

    def test_answers_resolve_questions_at_their_scope(self):
        extra = [W.namespace("N:ELM", "person", "ELM", ("HOSPITALS.ELM_CLEAN.VISITS", "CARDREF")),
                 W.namespace("N:CEDAR", "person", "CEDAR_MEDICAL_CAMP", ("HOSPITALS.CEDAR_MEDICAL_CAMP.CAMP_VISITS",
                                                                          "PATIENT_ID")),
                 W.namespace("N:ASPEN-VISITS", "person", "ASPEN", ("HOSPITALS.ASPEN_CLEAN.VISITS", "FOLIO_NO")),
                 Statement("F:CYPRESS", "facility", "CYPRESS", {"onboarded": True})]
        r = run("scenarios", W.Warehouse(self.objs), self.st + extra)
        A = r.beliefs.assessments
        for f in ("ELM_CLEAN.VISITS", "CEDAR_MEDICAL_CAMP.CAMP_VISITS", "ASPEN_CLEAN.VISITS"):
            self.assertEqual(A["HOSPITALS." + f].stage, "auto", f)     # stated: insufficient repetition is fine
        cyp = A["HOSPITALS.CYPRESS_CLEAN.VISITS"]                      # scope answered; meaning still open
        self.assertEqual(cyp.codes(), {"MEANING_UNRESOLVED"})
        # an answer about one facility does not reach another: Valley still asks
        self.assertIn("MEANING_UNRESOLVED", A["HOSPITALS.VALLEY_CLEAN.VISITS"].codes())

    def test_policies_stay_in_their_facility(self):
        hiv = self.A["HOSPITALS.OAKRIDGE_CLEAN.HIV_ART_DATA"]          # P-SEP is stated for MAPLE only
        self.assertIn("UNLINKED", hiv.codes())

    def test_only_aggregates_leave_the_warehouse(self):
        cells = set()
        for o in self.objs:
            for i, (c, t) in enumerate(o["columns"]):
                if t == "TEXT":
                    cells |= {str(r[i]) for r in o["rows"] if r[i] is not None}
        cells -= {"0", "1"}

        def walk(v):
            if isinstance(v, dict):
                for x in v.values():
                    yield from walk(x)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    yield from walk(x)
            else:
                yield v

        for o in self.r.observations:
            for v in walk(o.values):
                if isinstance(v, str):
                    self.assertTrue(re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) or re.fullmatch(r"[A9\W_]*", v), (o.kind, v))
                    self.assertNotIn(v, cells, (o.kind, v))
                else:
                    self.assertTrue(v is None or isinstance(v, (int, float)), (o.kind, v))

    def test_budget_is_respected_and_pending_is_never_accepted(self):
        r = understand(W.Warehouse(self.objs), self.st, now=W.NOW, budget=40000)
        self.assertLessEqual(r.cost, 40000)
        self.assertIn("pending", {a.stage for a in r.beliefs.assessments.values()})
        for a in r.beliefs.assessments.values():
            if a.stage in ("auto", "replacement"):
                self.assertFalse(any(g.blocking for g in a.gaps), a.fqn)

    def test_observations_are_reused_and_invalidated(self):
        r = understand(W.Warehouse(self.objs), self.st, now=W.NOW, observations=self.r.observations)
        self.assertEqual((len(r.measured), r.queries), (0, 0))
        changed = [dict(o, last_altered="2026-11-09 00:00:00") if o["name"] == "OPDVISITS" and o["schema"].startswith("MAPLE")
                   else o for o in self.objs]
        r = understand(W.Warehouse(changed), self.st, now=W.NOW, observations=self.r.observations)
        touched = {f for g in r.measured for f in g.subject}
        self.assertIn("HOSPITALS.MAPLE_HISTORICAL_DATA.OPDVISITS", touched)
        self.assertFalse({f for f in touched if not f.startswith("HOSPITALS.MAPLE")}, touched)
        self.assertEqual(r.beliefs.assessments["HOSPITALS.MAPLE_HISTORICAL_DATA.ADMISSIONS"].stage, "not_a_source")

    def test_snowflake_dialect_builds(self):
        """Builds only - the Snowflake dialect is not compiled here."""
        from patient_discovery.model.concepts import Gap
        cat = {o["fqn"]: o for o in self.wh.catalogue()}
        kinds = set()
        for g in (Gap(o.kind, o.subject, o.args) for o in self.r.observations):
            for sql in M.statements_for(M.Snowflake, cat, g):
                kinds.add(g.kind)
                self.assertNotIn("SELECT *", sql.upper())
                self.assertIn('"HOSPITALS"', sql)
        self.assertTrue({"screen", "profile", "pair", "shape", "containment", "align", "formats", "window"} <= kinds,
                        kinds)


class Replay(unittest.TestCase):
    """Each counted source held out in turn; only what the model may legitimately know is stated."""

    @classmethod
    def setUpClass(cls):
        objs = W.replica_world()
        cls.objs = objs
        cls.runs = {name: run("replica", W.Warehouse(objs), W.replica_statements(name)) for name in W.SOURCES}
        cls.full = run("replica", W.Warehouse(objs), W.replica_statements())
        if REPORT:
            print("\nleave-one-out replay")
            for name, r in cls.runs.items():
                a = r.beliefs.assessments[W.SOURCES[name].subject]
                print(f"  {name:15} {a.stage:9} codes={sorted(a.codes())}")

    def test_held_out_source_contributes_nothing(self):
        for name, held in W.SOURCES.items():
            st = W.replica_statements(name)
            with self.subTest(source=name):
                for s in st:
                    self.assertNotEqual(s.subject, held.subject, s.id)
                    for m in s.assertion.get("members", ()):
                        self.assertNotEqual(m[0], held.subject, s.id)
                others = {family(s.subject.split(".")[1]) for n, s in W.SOURCES.items() if n != name}
                self.assertEqual({s.subject for s in st if s.kind == "facility"}, others)
                seps = {s.subject for s in st if s.kind == "unlinked_populations"}
                self.assertEqual(seps, set() if name in ("Kakamega", "Kakamega HIV") else {"KAKAMEGA"})

    def test_expected_outcomes_and_reasons(self):
        for name, r in self.runs.items():
            stage, must, must_not, frags = W.EXPECTED_REPLAY[name]
            check_expected(self, r.beliefs.assessments, {W.SOURCES[name].subject: (stage, must, must_not, frags)})

    def test_nothing_is_accepted_and_held_out_meaning_stays_open(self):
        for name, r in self.runs.items():
            with self.subTest(source=name):
                a = r.beliefs.assessments[W.SOURCES[name].subject]
                self.assertTrue(a.selected is None or a.selected.meaning is None)
                self.assertEqual([f for f, x in r.beliefs.assessments.items() if x.stage in ("auto", "replacement")], [])

    def test_counted_sources_meaning_audit(self):
        counted = self.full.beliefs.counted
        for name, s in W.SOURCES.items():
            with self.subTest(source=name):
                self.assertEqual(counted[s.subject]["meaning"], W.EXPECTED_COUNTED_MEANING.get(name))


class OldSuitesCrossCheck(unittest.TestCase):
    """The existing scenario catalogue and 43 of the 45 round-2 checks through the model, every facility onboarded.
    A false acceptance = auto where the old expectation excludes 'proposed'. (Weak by construction: these worlds
    contain no namespace statements, so the model can accept nothing; kept as a guard, not as evidence.)"""

    @staticmethod
    def statements(fx, registry):
        from .fixtures import classified
        st = []
        for f, (status, src) in registry.items():
            if status == "active" and f in fx.objs:
                c = classified(fx.objs[f])
                st.append(W.source(f, c.key, c.record_key, c.deleted_column, c.version_column, c.received_column, src))
            elif status == "rejected":
                st.append(Statement(f"R:{f}", "rejected", f, {}, origin="registry"))
        return st + W.policies({family(o["schema"]) for o in fx.objs.values()})

    @classmethod
    def setUpClass(cls):
        from . import adversarial_round2 as adv
        from .cases import build
        from .fixtures import REGISTRY, SQLiteWarehouse
        cls.pairs = []
        cases = build()
        fx = SQLiteWarehouse([c["obj"] for c in cases])
        r = understand(W.FixtureAdapter(fx), cls.statements(fx, REGISTRY), now=W.NOW)
        for c in cases:
            f = f"HOSPITALS.{c['obj']['schema']}.{c['obj']['name']}"
            cls.pairs.append(("cases", c["id"], {c["status"]}, r.beliefs.assessments[f].stage))
        current = {}
        orig_world, orig_check = adv.world, adv.check

        def world(objs, **kw):
            wh, R = orig_world(objs, **kw)
            current["b"] = understand(W.FixtureAdapter(wh), cls.statements(wh, kw.get("registry", adv.REGISTRY)),
                                      now=W.NOW).beliefs
            return wh, R

        def check(cid, scenario, o, expected, severity, rows, note=""):
            cls.pairs.append(("round2", cid, set(expected), current["b"].assessments[adv.fqn(o)].stage))
            return orig_check(cid, scenario, o, expected, severity, rows, note)

        adv.world, adv.check = world, check
        try:
            for fn in adv.ALL:
                fn()
        finally:
            adv.world, adv.check = orig_world, orig_check
            adv.RESULTS.clear()

    def test_no_false_acceptance(self):
        bad = [(s, cid, exp, st) for s, cid, exp, st in self.pairs if st in ("auto", "replacement") and "proposed" not in exp]
        self.assertEqual(bad, [])

    def test_all_dataset_checks_ran(self):
        # N19 (malformed registry rows) and N25 (the old job's time budget) test the old job's mechanics
        self.assertEqual(sum(1 for p in self.pairs if p[0] == "round2"), 43)


if __name__ == "__main__":
    unittest.main()
