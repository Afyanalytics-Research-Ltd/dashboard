"""HMIS counts: which environment and rows are used, and how facilities join the patient counts table.

Run from analytics_app/dashboards/internal_reporting:  python -m unittest afya_metrics.test_hmis
"""
import unittest
from datetime import datetime

from afya_metrics import hmis as H

NOW = datetime(2026, 11, 12, 9, 0)
WAREHOUSE_SOURCES = ["Kisumu Specialist Hospital (KSH)", "Wema Medical", "Medhanit", "Portbell", "Silverwood",
                     "Tenri", "Lodwar"]


def row(which, org_id, org, fac_id, fac, n, scope="facility", at=datetime(2026, 11, 12, 8, 0),
        method="gateway_count"):
    return dict(WHICH=which, CAPTURED_AT=at, ORGANIZATION_ID=org_id, ORGANIZATION_NAME=org, FACILITY_ID=fac_id,
                FACILITY_NAME=fac, PATIENT_COUNT=n, SCOPE=scope, COUNT_METHOD=method)


# shaped like the migrate runs (facility and unassigned rows; organization rows are filtered in SQL)
FEED = [
    row("latest", 1, "Staging", 1, "Collabmed", 179334),
    row("latest", 13, "EAGLE NURSING HOME", 16, "Kangemi", 65849),
    row("latest", 4, "Kisumu Specialist", 4, "Ksh", 53291),
    row("latest", 6, "THE GYNO SOLUTIONS CENTER LTD", 8, "Gyno Solutions", 24547),
    row("latest", 16, "Medhanit", 20, "Medhanit", 2739),
    row("latest", 7, "Portbell", 9, "Portbell", 736),
    row("latest", 5, "Level6 Hospital", 7, "KSH", 170),
    row("latest", 13, "EAGLE NURSING HOME", None, "(unassigned)", 4, scope="unassigned"),
    row("latest", 20, "demo facility 6", 24, "demo facility 6", 3),
]
OWN_ROWS = 65849 + 24547 + 170 + 4          # Kangemi, Gyno, Level6's KSH, Eagle unassigned


def names(view):
    return {f["name"] for f in view["facilities"]}


class Environment(unittest.TestCase):
    def test_production_is_the_default_once_it_has_runs(self):
        self.assertEqual(H.pick_environment({"migrate": NOW, "production": NOW}), "production")

    def test_migrate_until_production_arrives(self):
        self.assertEqual(H.pick_environment({"migrate": NOW}), "migrate")
        self.assertIsNone(H.pick_environment({}))

    def test_unsafe_names_are_never_used_in_sql(self):
        self.assertIsNone(H.pick_environment({"production'; drop": NOW}))

    def test_switch_choices_resolve_to_stored_names(self):
        self.assertEqual(H.resolve("production", {"PRODUCTION": NOW, "migrate": NOW}), "PRODUCTION")
        self.assertIsNone(H.resolve("production", {"migrate": NOW}))


class Facilities(unittest.TestCase):
    def test_either_environment_feeds_the_table(self):
        for env in ("migrate", "production"):
            rows = H.counted_rows(H.build(FEED, env, WAREHOUSE_SOURCES, NOW))
            self.assertEqual(sum(r["UPDS"] for r in rows), OWN_ROWS, env)

    def test_test_organisations_never_appear(self):
        v = H.build(FEED, "migrate", WAREHOUSE_SOURCES, NOW)
        self.assertFalse({"Collabmed · Staging", "demo facility 6"} & names(v))

    def test_warehouse_facilities_are_not_added_twice(self):
        v = H.build(FEED, "production", WAREHOUSE_SOURCES, NOW)
        self.assertFalse({"Ksh · Kisumu Specialist", "Medhanit", "Portbell"} & names(v))
        self.assertEqual(v["on_warehouse"], {"Kisumu Specialist Hospital (KSH)": 53291, "Medhanit": 2739,
                                             "Portbell": 736})
        self.assertIn("KSH · Level6 Hospital", names(v))       # a different KSH: matched on the organisation too

    def test_unlisted_name_matching_a_warehouse_source_goes_on_that_row(self):
        v = H.build([row("latest", 30, "Tenri Hospital Group", 40, "Tenri", 900)], "production",
                    WAREHOUSE_SOURCES, NOW)
        self.assertEqual((v["facilities"], v["on_warehouse"]), ([], {"Tenri": 900}))

    def test_rows_add_patients_to_upds_only(self):
        rows = H.counted_rows(H.build(FEED, "production", WAREHOUSE_SOURCES, NOW))
        self.assertTrue(all(r["HMIS"] and r["PATIENT_RECORDS"] == 0 for r in rows))

    def test_other_methods_and_zero_counts_are_skipped(self):
        v = H.build([row("latest", 13, "EAGLE NURSING HOME", 16, "Kangemi", 10, method="estimate"),
                     row("latest", 13, "EAGLE NURSING HOME", 17, "Kibra", 0)], "production", [], NOW)
        self.assertEqual((v["facilities"], v["dropped"]), ([], 2))


class Flags(unittest.TestCase):
    def test_jump_since_the_previous_day(self):
        feed = [row("latest", 13, "EAGLE NURSING HOME", 16, "Kangemi", 70000),
                row("previous", 13, "EAGLE NURSING HOME", 16, "Kangemi", 50000, at=datetime(2026, 11, 11, 8, 0)),
                row("latest", 13, "EAGLE NURSING HOME", 17, "Kibra", 18900),
                row("previous", 13, "EAGLE NURSING HOME", 17, "Kibra", 18817, at=datetime(2026, 11, 11, 8, 0))]
        rows = {r["SOURCE"]: r for r in H.counted_rows(H.build(feed, "production", [], NOW))}
        self.assertEqual(rows["Kangemi · Eagle Nursing Home"]["FLAGS"], ["+40% since 11 Nov"])
        self.assertEqual(rows["Kibra · Eagle Nursing Home"]["FLAGS"], [])

    def test_stale_run(self):
        old = [row("latest", 13, "EAGLE NURSING HOME", 16, "Kangemi", 1, at=datetime(2026, 11, 10, 8, 0))]
        self.assertTrue(H.build(old, "production", [], NOW)["stale"])
        self.assertFalse(H.build(FEED, "production", [], NOW)["stale"])


if __name__ == "__main__":
    unittest.main()
