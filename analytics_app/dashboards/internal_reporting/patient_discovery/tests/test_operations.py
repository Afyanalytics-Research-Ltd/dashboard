"""Operational safety of the authoritative path: question pack and answers, privacy of measurements, cache
invalidation, failure handling, the SQL guard. Synthetic data only."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from patient_discovery.model import measure as M
from patient_discovery.model import questions as Q
from patient_discovery.model import statements as S
from patient_discovery.model.concepts import Gap
from patient_discovery.model.planner import understand
from patient_discovery.model.snowflake import check

from . import model_world as W


class QuestionsAndAnswers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.objs = W.replica_world()
        cls.st = W.replica_statements()
        cls.r = understand(W.Warehouse(cls.objs), cls.st, now=W.NOW)

    def test_counted_sources_are_grouped_once_per_facility_and_column(self):
        groups = Q.build(self.r.beliefs)
        meaning = [g for g in groups if g["type"] == "meaning"]
        keys = [(g["facility"], g["column"]) for g in meaning]
        self.assertEqual(len(keys), len(set(keys)))
        counted = {m[0] for g in meaning for m in g["members"]}
        for name, s in W.SOURCES.items():
            if name in W.EXPECTED_COUNTED_MEANING:
                self.assertNotIn(s.subject, counted, name)          # meaning already known: never asked again
            else:
                self.assertIn(s.subject, counted, name)

    def test_answers_become_scoped_statements_and_stop_the_question(self):
        groups = Q.build(self.r.beliefs)
        g = next(g for g in groups if g["type"] == "meaning" and g["facility"] == "KISUMU")
        g["answer"] = "person"
        with tempfile.TemporaryDirectory() as d:
            pack, store = Path(d, "q.json"), Path(d, "statements.json")
            Q.write(groups, pack)
            new = Q.record(pack, "tester", path=store)
            self.assertEqual(len(new), 1)
            s = new[0]
            self.assertEqual((s["kind"], s["assertion"]["meaning"], s["assertion"]["scope"]),
                             ("namespace", "person", {"facility": "KISUMU"}))
            self.assertEqual(s["assertion"]["provenance"]["question"], g["id"])
            Q.record(pack, "tester", path=store)                    # recording twice adds nothing new
            self.assertEqual(len(json.loads(store.read_text())["statements"]), 1)
            st = self.st + S.read_file(store)
            r = understand(W.Warehouse(self.objs), st, now=W.NOW, observations=self.r.observations)
            self.assertNotIn(g["id"], {x["id"] for x in Q.build(r.beliefs)})
            ksh = r.beliefs.counted["HOSPITALS.KISUMU_V3_CLEAN.EVALUATION_VISITS"]
            self.assertEqual(ksh["meaning"], "person")

    def test_questions_close_when_all_their_tables_are_not_sources(self):
        a, b = "HOSPITALS.F_CLEAN.PAYMENTS", "HOSPITALS.F_CLEAN.VISITS"
        meaning = dict(id="Q-1", type="meaning", facility="F", members=[[a, "PATIENT_ID"]])
        both = dict(id="Q-2", type="GRAIN", facility="F", objects=[a, b])
        scope = dict(id="Q-3", type="scope", facility="F", objects=[a, b])
        out = [dict(id=f"X:{a}", kind="not_a_source", subject=a, assertion={})]
        self.assertTrue(Q.settled(meaning, out))
        self.assertFalse(Q.settled(both, out))                  # VISITS is still undecided
        self.assertFalse(Q.settled(scope, out))                 # the facility still has VISITS to count
        self.assertTrue(Q.settled(scope, out + [dict(id=f"X:{b}", kind="not_a_source", subject=b, assertion={})]))
        self.assertFalse(Q.settled(meaning, []))

    def test_legacy_proposals_are_never_statements(self):
        st, legacy = S.from_registry([
            dict(SOURCE="x", SOURCE_OBJECT="HOSPITALS.A.B", UPD_KEY="K", RECORD_KEY=None, DELETED_COLUMN=None,
                 VERSION_COLUMN=None, RECEIVED_COLUMN=None, STATUS="proposed")])
        self.assertEqual((st, len(legacy)), ([], 1))


class Registration(unittest.TestCase):
    """Answer -> immediate effect: outcomes precomputed by the run, registration once every required answer is in."""

    @classmethod
    def setUpClass(cls):
        objs, cls.st = W.scenario_world()
        wh = W.Warehouse(objs)
        r = understand(wh, cls.st, now=W.NOW)
        cls.groups = Q.with_outcomes(Q.build(r.beliefs), wh.catalogue(), r.observations, cls.st, W.NOW)
        cls.q = {(g["type"], g["facility"]): g for g in cls.groups}

    def due(self, statements, registry=()):
        return Q.due_registrations(self.groups, statements, list(registry))

    def test_outcomes_are_precomputed(self):
        out = self.q[("meaning", "VALLEY")]["on_person"]["HOSPITALS.VALLEY_CLEAN.VISITS"]
        self.assertEqual((out["stage"], out["config"]["key"], out["config"]["record"]), ("auto", "PATIENT_ID", "VISIT_ID"))

    def test_an_answer_registers_at_once(self):
        g = self.q[("meaning", "VALLEY")]
        self.assertEqual(self.due([]), [])
        st = Q.to_statements(g, "person", "tester", "clinic records", {"dashboard_name": "Valley Clinic"})
        due = self.due(st)
        self.assertEqual([(r["object"], r["name"], r["config"]["key"]) for r in due],
                         [("HOSPITALS.VALLEY_CLEAN.VISITS", "Valley Clinic", "PATIENT_ID")])
        self.assertEqual((due[0]["provenance"]["question"], due[0]["provenance"]["by"]), (g["id"], "tester"))
        self.assertEqual(self.due(Q.to_statements(g, "household", "tester")), [])

    def test_scope_must_be_answered_too(self):
        g, s = self.q[("meaning", "CYPRESS")], self.q[("scope", "CYPRESS")]
        meaning = Q.to_statements(g, "person", "tester")
        self.assertEqual(self.due(meaning), [])
        self.assertEqual(self.due(meaning + Q.to_statements(s, "no", "tester")), [])
        due = self.due(meaning + Q.to_statements(s, "yes", "tester"))
        self.assertEqual([r["object"] for r in due], ["HOSPITALS.CYPRESS_CLEAN.VISITS"])

    def test_versions_and_open_questions_are_never_registered(self):
        g = self.q[("meaning", "KESTREL")]
        self.assertNotEqual(g["on_person"]["HOSPITALS.KESTREL_V2_CLEAN.VISITS"]["stage"], "auto")
        self.assertEqual(self.due(Q.to_statements(g, "person", "tester")), [])

    def test_no_duplicates_or_name_clashes(self):
        st = Q.to_statements(self.q[("meaning", "VALLEY")], "person", "tester", "", {"dashboard_name": "Valley"})
        self.assertEqual(self.due(st, [{"SOURCE": "x", "SOURCE_OBJECT": "hospitals.valley_clean.visits "}]), [])
        self.assertEqual(self.due(st, [{"SOURCE": "Valley", "SOURCE_OBJECT": "HOSPITALS.OTHER.T"}]), [])

    def test_outcomes_needing_measurements_are_measured_before_anyone_answers(self):
        objs, st = W.scenario_world()
        wh = W.Warehouse(objs)
        first = understand(wh, st, now=W.NOW, budget=60000)                  # partial evidence
        groups = Q.build(first.beliefs)
        obs = first.observations
        for g in groups:                                                     # what cmd_profile does
            hyp = Q.hypothesis(g, st)
            if hyp is not None:
                obs = understand(wh, hyp, now=W.NOW, observations=obs).observations
        groups = Q.with_outcomes(groups, wh.catalogue(), obs, st, W.NOW)
        pending = [(g["facility"], f) for g in groups for f, o in g.get("on_person", {}).items()
                   if o["stage"] == "pending"]
        self.assertEqual(pending, [])

    def test_registry_writes_are_insert_if_absent_only(self):
        from patient_discovery.model import store
        st = Q.to_statements(self.q[("meaning", "VALLEY")], "person", "tester")
        reg = self.due(st)[0]
        store.check_writes([store.register_sql(reg)] + store.record_statements([store.registered_statement(reg)]))
        R = store.REGISTRY
        for bad in (f"UPDATE {R} SET STATUS = 'active'", f"DELETE FROM {R} WHERE SOURCE = 'x'",
                    f"INSERT INTO {R} (SOURCE, SOURCE_OBJECT, UPD_KEY) VALUES ('a', 'b', 'c')",
                    f"INSERT INTO {R} (SOURCE, SOURCE_OBJECT, UPD_KEY) SELECT 1, 2, 3 WHERE NOT EXISTS "
                    f"(SELECT 1 FROM {R}); UPDATE {R} SET STATUS = 'x'",
                    "INSERT INTO HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNTS VALUES (1)"):
            with self.assertRaises(PermissionError, msg=bad):
                store.check_writes([(bad, None)])


class DiscoveryStore(unittest.TestCase):
    def test_answering_again_adds_members_and_keeps_history(self):
        from patient_discovery.model import store
        q = dict(id="Q-1", type="meaning", facility="F", column="PID", members=[["HOSPITALS.F.B", "PID"]])
        old = Q.to_statements(dict(q, members=[["HOSPITALS.F.A", "PID"]]), "person", "ann", "first")[0]
        new = store.merge(Q.to_statements(q, "person", "bob", "second"), {old["id"]: old})
        self.assertEqual(new[0]["assertion"]["members"], [["HOSPITALS.F.A", "PID"], ["HOSPITALS.F.B", "PID"]])
        self.assertEqual(new[0]["assertion"]["history"][0]["by"], "ann")
        self.assertEqual(store.merge([old], {old["id"]: old}), [])          # the same fact is not recorded twice
        sql, params = store.record_statements(new)[0]
        self.assertIn(store.STATEMENTS, sql)
        self.assertEqual((params[0], params[4], params[5]), ("N:F:PID", "Q-1", "bob"))

    def test_discovery_writes_only_its_two_tables(self):
        from patient_discovery.model import store
        from patient_discovery.model.snowflake import Snowflake

        class Cur:
            def __init__(self):
                self.sql = []

            def execute(self, sql, params=None):
                self.sql.append(sql)

            def close(self):
                pass

        cur = Cur()
        wh = Snowflake.__new__(Snowflake)
        wh.conn = type("Conn", (), {"cursor": lambda self: cur})()
        for bad in ("INSERT INTO HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_SOURCES VALUES (1)",
                    "DELETE FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNTS",
                    f"DELETE FROM {store.QUESTIONS}; DROP TABLE x", "UPDATE x SET a = 1"):
            with self.assertRaises(PermissionError, msg=bad):
                wh.write([(bad, None)])
        self.assertEqual(cur.sql, [])
        wh.write(store.publish_questions([dict(id="Q-1", type="scope", facility="F")]) +
                 [(sql, None) for sql in store.DDL])
        self.assertEqual((cur.sql[0], cur.sql[-1]), ("BEGIN", "COMMIT"))


class MeasurementSafety(unittest.TestCase):
    def test_row_hash_and_visit_date_skip_personal_columns(self):
        cols = [("PATIENT_ID", "TEXT"), ("PATIENT_NAME", "TEXT"), ("PHONE_NO", "TEXT"), ("BIRTHDATE", "DATE"),
                ("DIAGNOSIS", "TEXT")]
        sql = M.screen_sql(M.Snowflake, "HOSPITALS.X.Y", [("PATIENT_ID", "TEXT")], None, None, cols)[0]
        self.assertNotIn("HASH(*)", sql)
        for c in ("PATIENT_NAME", "PHONE_NO", "BIRTHDATE"):
            self.assertNotIn(f'"{c}"', sql)
        o = W.obj("PRIV_CLEAN", "VISITS", [("VISIT_ID", "TEXT"), ("PATIENT_ID", "TEXT"), ("BIRTHDATE", "DATE")],
                  [("v1", "p1", "1990-01-01")])
        r = understand(W.Warehouse([o]), [], now=W.NOW, budget=0)
        self.assertIsNone(r.beliefs.assessments["HOSPITALS.PRIV_CLEAN.VISITS"].grain.date)

    def test_guard_refuses_writes_and_star(self):
        for bad in ("SELECT * FROM t", "DELETE FROM t", "SELECT 1; DROP TABLE t", "CALL p()",
                    "SELECT a.* FROM t a"):
            with self.assertRaises(ValueError, msg=bad):
                check(bad)
        check('SELECT COUNT(*), COUNT(DISTINCT "UPDATED_AT") FROM "HOSPITALS"."X"."Y"')
        check("SELECT COUNT(DISTINCT IFF(REGEXP_LIKE(x, '.*[A-Z].*'), 'A', '')) FROM t")   # .* inside a literal
        for bad in ("SELECT 'a', * FROM t", "SELECT 'x'; DROP TABLE t", "SELECT 'it''s', t.* FROM t"):
            with self.assertRaises(ValueError, msg=bad):
                check(bad)


class CacheAndFailures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.objs, cls.st = W.scenario_world()
        cls.r = understand(W.Warehouse(cls.objs), cls.st, now=W.NOW)

    def test_view_observations_expire(self):
        later = W.NOW + timedelta(days=40)
        r = understand(W.Warehouse(self.objs), self.st, now=later, observations=self.r.observations)
        views = {o["fqn"] for o in W.Warehouse(self.objs).catalogue() if o["type"] != "BASE TABLE"}
        before = {(o.kind, o.subject, o.args) for o in self.r.observations}
        again = [(g.kind, g.subject, g.args) for g in r.measured if (g.kind, g.subject, g.args) in before]
        self.assertTrue(any(set(k[1]) & views for k in again))                   # view observations expired
        self.assertFalse([k for k in again if not set(k[1]) & views])             # table observations reused

    def test_a_raw_store_change_invalidates_its_views(self):
        store = W.obj("ACME_RAW", "EVENTS_RAW", [("PAYLOAD", "TEXT")], [("x",)])
        view = W.obj("ACME_CLEAN", "VISITS", W.EV_COLS, [], "VIEW", refs=["HOSPITALS.ACME_RAW.EVENTS_RAW"])
        mark = lambda objs: [dict(o, store=True) if o["schema"] == "ACME_RAW" else o for o in objs]
        r1 = understand(W.Warehouse(mark([store, view])), [], now=W.NOW)
        changed = dict(store, last_altered="2026-11-09 00:00:00", rows=[("x",), ("y",)], row_count=2)
        r2 = understand(W.Warehouse(mark([changed, view])), [], now=W.NOW, observations=r1.observations)
        self.assertIn("HOSPITALS.ACME_CLEAN.VISITS", {f for g in r2.measured for f in g.subject})

    def test_a_failing_measurement_keeps_the_rest(self):
        wh = W.Warehouse(self.objs)
        real = wh.query
        def query(sql):
            if "HILLSIDE" in sql:
                raise RuntimeError("Numeric value 'H00001' is not recognized")
            return real(sql)
        wh.query = query
        r = understand(wh, self.st, now=W.NOW)
        self.assertTrue(r.failed)
        self.assertNotIn("H00001", " ".join(e for _, e in r.failed))
        A = r.beliefs.assessments
        self.assertEqual(A["HOSPITALS.HILLSIDE_CLEAN.VISITS"].stage, "pending")
        self.assertEqual(A["HOSPITALS.WREN_CLEAN.VISITS"].stage, "auto")

    def test_dry_run_never_queries(self):
        wh = W.Warehouse(self.objs + [W.obj("EMPTY_CLEAN", "VISITS", W.EV_COLS, [])])   # zero-cost gap included
        r = understand(wh, self.st, now=W.NOW, dry_run=True)
        self.assertEqual((r.queries, len(r.measured)), (0, 0))
        self.assertTrue(r.unmet)

    def test_no_link_checks_once_a_tables_meaning_is_set(self):
        derivation = ("derivation by records", "lineage values", "namespace formats")
        for objs, st in ((self.objs, self.st), (W.replica_world(), W.replica_statements())):
            r = understand(W.Warehouse(objs), st, now=W.NOW)
            stated = {f for s in st if s.kind == "namespace" for f, _ in s.assertion.get("members", [])}
            # a link check between two tables that both already have a stated meaning can only waste statements
            asked = [g for g in r.measured + list(r.unmet) if g.why in derivation and set(g.subject) <= stated]
            self.assertTrue(stated)
            self.assertFalse(asked, [(g.subject, g.why) for g in asked][:3])

    def test_statement_limit_is_enforced(self):
        wh = W.Warehouse(self.objs)
        r = understand(wh, self.st, now=W.NOW, max_statements=25)
        self.assertLessEqual(r.queries, 25)


if __name__ == "__main__":
    unittest.main()
