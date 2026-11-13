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
# Kangemi, Gyno, Level6's KSH, and Medhanit, KSH and Portbell (the HMIS replaces the warehouse for these)
OWN_ROWS = 65849 + 24547 + 170 + 2739 + 53291 + 736


def names(view):
    return {f["name"] for f in view["facilities"]}


class Environment(unittest.TestCase):
    PARTS = {"Production": (H.build(FEED, "prod", WAREHOUSE_SOURCES, NOW), []),
             "Migrate": (H.build(FEED, "migrate", WAREHOUSE_SOURCES, NOW), [])}

    def test_filter_options_and_default(self):
        self.assertEqual(H.env_options(self.PARTS), ["Warehouse", "Production", "Migrate"])
        self.assertEqual(H.default_envs(self.PARTS), ["Warehouse", "Production"])
        self.assertEqual(H.default_envs({"Migrate": self.PARTS["Migrate"]}), ["Warehouse", "Migrate"])
        self.assertEqual(H.default_envs({}), ["Warehouse"])

    def test_selection_drives_rows_and_replacement(self):
        wh = [{"SOURCE": "Medhanit", "PATIENT_RECORDS": 913, "UPDS": 551},
              {"SOURCE": "Tenri", "PATIENT_RECORDS": 10, "UPDS": 5}]
        rows, _, overlap = H.combine(wh, [], self.PARTS, ["Warehouse"])
        self.assertEqual({r["SOURCE"] for r in rows}, {"Tenri"})       # the HMIS is Medhanit's only source
        rows, _, _ = H.combine(wh, [], {}, ["Warehouse"])
        self.assertEqual({r["SOURCE"] for r in rows}, {"Medhanit", "Tenri"})      # HMIS not loaded: warehouse stays
        rows, _, overlap = H.combine(wh, [], self.PARTS, ["Warehouse", "Production"])
        med = [r for r in rows if r["SOURCE"] == "Medhanit"]
        self.assertEqual([(r["ENV"], r["UPDS"]) for r in med], [("Production", 2739)])   # HMIS replaces it
        self.assertEqual(overlap, [])
        rows, _, overlap = H.combine(wh, [], self.PARTS, ["Production", "Migrate"])
        self.assertEqual(overlap, ["Production", "Migrate"])
        self.assertIn("Medhanit (Migrate)", {r["SOURCE"] for r in rows})          # same-named facilities kept apart
        self.assertFalse(any(r["ENV"] == "Warehouse" for r in rows))

    def test_unsafe_names_are_never_used_in_sql(self):
        self.assertIsNone(H.resolve("prod", {"prod'; drop": NOW}))

    def test_switch_choices_resolve_to_stored_names(self):
        self.assertEqual(H.resolve("prod", {"prod": NOW, "migrate": NOW}), "prod")
        self.assertEqual(H.resolve("prod", {"PRODUCTION": NOW}), "PRODUCTION")
        self.assertIsNone(H.resolve("prod", {"migrate": NOW}))


class Facilities(unittest.TestCase):
    def test_either_environment_feeds_the_table(self):
        for env in ("migrate", "production"):
            rows = H.counted_rows(H.build(FEED, env, WAREHOUSE_SOURCES, NOW))
            self.assertEqual(sum(r["UPDS"] for r in rows), OWN_ROWS, env)

    def test_test_organisations_never_appear(self):
        v = H.build(FEED, "migrate", WAREHOUSE_SOURCES, NOW)
        self.assertFalse({"Collabmed · Staging", "demo facility 6"} & names(v))

    def test_hmis_replaces_matching_warehouse_sources_under_its_own_names(self):
        v = H.build(FEED, "production", WAREHOUSE_SOURCES, NOW)
        self.assertEqual(v["replaced"], {"Kisumu Specialist Hospital (KSH)", "Portbell", "Medhanit"})
        self.assertTrue({"Ksh", "Portbell", "Medhanit"} <= names(v))   # HMIS facility names, not warehouse sources
        self.assertEqual(v["on_warehouse"], {})
        self.assertIn("KSH · Level6 Hospital", names(v))       # a different KSH: matched on the organisation too

    def test_demo_and_probe_set_ups_never_appear(self):
        feed = [row("latest", 1, "Hill Wood Health Care Limited", 1, "HILL WOOD HEALTH CARE LIMITED", 326)] + [
            row("latest", o, name, o, fac, 0) for o, name, fac in (
                (6, "Demo Facility", "HILL WOOD HEALTH CARE LIMITED"), (15, "demo9", "HILL WOOD HEALTH CARE LIMITED"),
                (16, "demo 2", "HILL WOOD HEALTH CARE LIMITED"), (17, "demo", "HILL WOOD HEALTH CARE LIMITED"),
                (14, "demo-probe2-1789299052", "Demo Facility Probe2"), (40, "Demola Clinic", "Demola Clinic"))]
        self.assertEqual(sorted(names(H.build(feed, "prod", [], NOW))),
                         ["Demola Clinic", "Hill Wood Health Care Limited"])

    def test_a_facility_registered_more_than_once_is_listed_once(self):
        feed = [row("latest", 54, "Lofty", 53, "PROGRESSIVE LOFTY HEALTH CENTRE", 81),       # most patients
                row("latest", 26, "Lofty", 25, "Progressive  Lofty Health Centre", 7),
                row("latest", 113, "Kegonga", 111, "KEGONGA MEDICAL CENTER", 75),
                row("latest", 70, "Zen", 111, "KEGONGA MEDICAL CENTER", 2),
                row("latest", 20, "Kegonga", 19, "KEGONGA MEDICAL CENTER", 0),
                row("latest", 145, "Aroha", 143, "AROHA CANCER CENTRE LIMITED", 7),          # tie: newest IDs
                row("latest", 100, "Kaptono", 143, "AROHA CANCER CENTRE LIMITED", 7),
                row("latest", 296, "Ebenezzer", 295, "EBENEZZER MEDICAL CENTRE", 0),         # both empty: newest
                row("latest", 297, "Ebenezzer", 296, "EBENEZZER MEDICAL CENTRE", 0),
                row("latest", 66, "Afyastar", 64, "AFYASTAR MEDICAL CENTRE LTD", 412),
                row("latest", 4, "Kisumu Specialist", 4, "Ksh", 50),                         # short names: never grouped
                row("latest", 5, "Level6 Hospital", 7, "KSH", 170)]
        v = H.build(feed, "prod", [], NOW)
        self.assertEqual(sorted(f["ids"] for f in v["facilities"]),
                         [(4, 4), (5, 7), (54, 53), (66, 64), (113, 111), (145, 143), (297, 296)])
        self.assertEqual(v["merged"], 5)

    def test_rows_without_a_facility_id_are_dropped(self):
        feed = FEED + [row("latest", 30, "Other", None, "Somewhere", 6)]
        v = H.build(feed, "migrate", WAREHOUSE_SOURCES, NOW)
        self.assertFalse([f for f in v["facilities"] if "unassigned" in f["name"].lower() or "Somewhere" in f["name"]])


class Production(unittest.TestCase):
    """Production posts no organisation names and few facility names: exceptions and labels go by ID."""
    PROD = [
        row("latest", 199, None, 329, None, 65849),
        row("latest", 284, None, 283, "KISUMU SPECIALISTS HOSPITAL", 53771),
        row("latest", 250, None, 247, "Kisumu Specialists Hospital", 52829),
        row("latest", 159, None, 157, "THE GYNO SOLUTIONS CENTER LIMITED", 24681),
        row("latest", 258, None, 256, "WEMA MEDKIN MEDICAL CENTER LTD", 5342),
        row("latest", 53, None, 52, None, 2265),
        row("latest", 199, None, None, "(unassigned)", 4, scope="unassigned"),
        row("latest", 12, None, None, "(unassigned)", 2506, scope="unassigned"),
    ]

    def setUp(self):
        self.v = H.build(self.PROD, "prod", WAREHOUSE_SOURCES, NOW)
        self.rows = {r["SOURCE"]: r["UPDS"] for r in H.counted_rows(self.v)}

    def test_hmis_replaces_the_warehouse_for_medhanit(self):
        feed = self.PROD + [dict(row("latest", 71, None, 69, "Medhanit Healthcare Limited", 234), VISIT_COUNT=425)]
        v = H.build(feed, "prod", WAREHOUSE_SOURCES, NOW)
        rows = {r["SOURCE"]: r for r in H.counted_rows(v)}
        self.assertIn("Medhanit", v["replaced"])
        med = rows["Medhanit Healthcare Limited"]
        self.assertEqual((med["UPDS"], med["PATIENT_RECORDS"]), (234, 425))
        self.assertTrue(med["VISITS_POSTED"])
        self.assertNotIn("Medhanit", v["on_warehouse"])

    def test_ksh_comes_from_the_live_organisation_only(self):
        self.assertEqual(self.rows["Kisumu Specialists Hospital"], 53771)    # organisation 284, HMIS name
        self.assertNotIn(52829, self.rows.values())                          # 250 is a duplicate: never counted
        self.assertEqual(self.v["replaced"], {"Kisumu Specialist Hospital (KSH)", "Wema Medical"})

    def test_wema_comes_from_the_hmis(self):
        self.assertEqual(self.v["on_warehouse"], {})
        self.assertEqual(self.rows["Wema Medkin Medical Center Ltd"], 5342)

    def test_portbell_and_silverwood_come_from_the_hmis(self):
        feed = [row("latest", 263, None, 261, "PORTBELL MEDICAL CENTRE LIMITED", 897),
                row("latest", 82, None, 80, "SILVERWOOD MEDICAL CENTRE LTD", 733)]
        v = H.build(feed, "prod", WAREHOUSE_SOURCES, NOW)
        self.assertEqual(v["replaced"], {"Portbell", "Silverwood"})
        self.assertEqual({r["SOURCE"]: r["UPDS"] for r in H.counted_rows(v)},
                         {"Portbell Medical Centre Limited": 897, "Silverwood Medical Centre Ltd": 733})

    def test_nameless_facilities_get_a_label(self):
        self.assertEqual(self.rows["Kangemi · Eagle Nursing Home"], 65849)
        self.assertEqual(self.rows["The Gyno Solutions Center Limited"], 24681)   # no stray separator
        self.assertEqual(self.rows["HMIS facility 52 (organisation 53)"], 2265)
        self.assertEqual(sum(self.rows.values()), 65849 + 24681 + 2265 + 53771 + 5342)   # unassigned rows dropped

    def test_unlisted_name_matching_a_warehouse_source_goes_on_that_row(self):
        v = H.build([row("latest", 30, "Tenri Hospital Group", 40, "Tenri", 900)], "production",
                    WAREHOUSE_SOURCES, NOW)
        self.assertEqual((v["facilities"], v["on_warehouse"]), ([], {"Tenri": 900}))

    def test_rows_add_patients_to_upds_only(self):
        rows = H.counted_rows(H.build(FEED, "production", WAREHOUSE_SOURCES, NOW))
        self.assertTrue(all(r["HMIS"] and r["PATIENT_RECORDS"] == 0 for r in rows))

    def test_other_methods_are_skipped(self):
        v = H.build([row("latest", 13, "EAGLE NURSING HOME", 16, "Kangemi", 10, method="estimate")], "production", [],
                    NOW)
        self.assertEqual((v["facilities"], v["dropped"]), ([], 1))

    def test_facilities_without_patients_are_listed_with_zero(self):
        v = H.build([row("latest", 13, "EAGLE NURSING HOME", 17, "Kibra", 0)], "production", [], NOW)
        rows = H.counted_rows(v)
        self.assertEqual([(r["SOURCE"], r["UPDS"]) for r in rows], [("Kibra · Eagle Nursing Home", 0)])
        self.assertEqual(v["dropped"], 0)


class DatedArrivals(unittest.TestCase):
    def test_daily_changes_become_arrivals_that_add_up_to_the_total(self):
        from datetime import date
        v = H.build(FEED, "migrate", WAREHOUSE_SOURCES, NOW)
        daily = [dict(ORGANIZATION_ID=13, FACILITY_ID=16, DATE=date(2026, 11, 11), ADDED=65000),   # first day: all
                 dict(ORGANIZATION_ID=13, FACILITY_ID=16, DATE=date(2026, 11, 12), ADDED=849),
                 dict(ORGANIZATION_ID=7, FACILITY_ID=9, DATE=date(2026, 11, 11), ADDED=736),      # Portbell: HMIS wins
                 dict(ORGANIZATION_ID=1, FACILITY_ID=1, DATE=date(2026, 11, 11), ADDED=179334),   # test organisation
                 dict(ORGANIZATION_ID=6, FACILITY_ID=8, DATE=date(2026, 11, 12), ADDED=0)]        # no change: omitted
        got = H.receipts(v, daily)
        self.assertEqual([(r["SOURCE"], r["DATE"].day, r["UPDS"], r["RECORDS"]) for r in got],
                         [("Kangemi · Eagle Nursing Home", 11, 65000, 0), ("Kangemi · Eagle Nursing Home", 12, 849, 0),
                          ("Portbell", 11, 736, 0)])
        self.assertEqual(sum(r["UPDS"] for r in got), 65849 + 736)


class Flags(unittest.TestCase):
    def test_jump_since_the_previous_day(self):
        feed = [row("latest", 13, "EAGLE NURSING HOME", 16, "Kangemi", 70000),
                row("previous", 13, "EAGLE NURSING HOME", 16, "Kangemi", 50000, at=datetime(2026, 11, 11, 8, 0)),
                row("latest", 13, "EAGLE NURSING HOME", 17, "Kibra", 18900),
                row("previous", 13, "EAGLE NURSING HOME", 17, "Kibra", 18817, at=datetime(2026, 11, 11, 8, 0))]
        rows = {r["SOURCE"]: r for r in H.counted_rows(H.build(feed, "production", [], NOW))}
        self.assertEqual(rows["Kangemi · Eagle Nursing Home"]["FLAGS"], ["+40% since 11 Nov"])
        self.assertEqual(rows["Kibra · Eagle Nursing Home"]["FLAGS"], [])

    def test_times_are_east_africa_time(self):
        from datetime import timezone
        utc = datetime.now(timezone.utc).replace(tzinfo=None)
        self.assertAlmostEqual((H.now_eat() - utc).total_seconds(), 3 * 3600, delta=5)

    def test_stale_run(self):
        old = [row("latest", 13, "EAGLE NURSING HOME", 16, "Kangemi", 1, at=datetime(2026, 11, 10, 8, 0))]
        self.assertTrue(H.build(old, "production", [], NOW)["stale"])
        self.assertFalse(H.build(FEED, "production", [], NOW)["stale"])


if __name__ == "__main__":
    unittest.main()
