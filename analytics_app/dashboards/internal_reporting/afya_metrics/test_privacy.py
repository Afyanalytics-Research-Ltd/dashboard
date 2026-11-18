"""Facility names behind codes on the Patient counts tab.

Run from analytics_app/dashboards/internal_reporting:  python -m unittest afya_metrics.test_privacy
"""
import unittest
from datetime import date

from afya_metrics import privacy as P

ROWS = [
    {"SOURCE": "Medhanit Healthcare Limited", "HMIS": True, "ENV": "Production", "ORGANIZATION_ID": 71, "FACILITY_ID": 69},
    {"SOURCE": "Kegonga Medical Center · Kegonga", "HMIS": True, "ENV": "Production", "ORGANIZATION_ID": 113,
     "FACILITY_ID": 111},
    {"SOURCE": "Kegonga Medical Center · Zen", "HMIS": True, "ENV": "Production", "ORGANIZATION_ID": 70,
     "FACILITY_ID": 111},
    {"SOURCE": "Tenri", "ENV": "Warehouse"},
    {"SOURCE": "Nandi", "ENV": "Warehouse"},
]


class Switch(unittest.TestCase):
    def test_hidden_unless_explicitly_off(self):
        for value in ("true", "TRUE", "1", "yes", "anything", ""):
            self.assertTrue(P.hidden(value), value)
        for value in ("false", "FALSE", " off ", "0", "no"):
            self.assertFalse(P.hidden(value), value)

    def test_the_deployment_variable_decides_and_unset_means_hidden(self):
        from unittest import mock
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertTrue(P.hidden())
        with mock.patch.dict("os.environ", {P.SWITCH: "false"}):
            self.assertFalse(P.hidden())
        with mock.patch.dict("os.environ", {P.SWITCH: "flase"}):              # a typo keeps names hidden
            self.assertTrue(P.hidden())


class Codes(unittest.TestCase):
    def test_codes_are_stable_and_never_show_a_name(self):
        rows, _ = P.apply(ROWS, [])
        shown = [r["SOURCE"] for r in rows]
        self.assertEqual(shown[0], "Facility F-0069")
        self.assertEqual(shown[1:3], ["Facility F-0111/O-0113", "Facility F-0111/O-0070"])
        self.assertTrue(shown[3].startswith("Source S-") and shown[4].startswith("Source S-"))
        self.assertEqual(len(set(shown)), len(shown))
        self.assertEqual(P.apply(ROWS, [])[0], rows)                       # same codes on every load
        for name in ("Medhanit", "Kegonga", "Tenri", "Nandi"):
            self.assertFalse(any(name.lower() in s.lower() for s in shown), name)

    def test_arrivals_follow_their_source(self):
        receipts = [{"SOURCE": "Tenri", "DATE": date(2026, 4, 2), "RECORDS": 5, "UPDS": 3},
                    {"SOURCE": "Medhanit Healthcare Limited", "DATE": date(2026, 11, 12), "RECORDS": 0, "UPDS": 2744},
                    {"SOURCE": "Lodwar", "DATE": None, "RECORDS": 1, "UPDS": 1}]          # not in the table
        rows, got = P.apply(ROWS, receipts)
        codes = {r["SOURCE"] for r in rows}
        self.assertTrue(got[0]["SOURCE"] in codes and got[1]["SOURCE"] == "Facility F-0069")
        self.assertTrue(got[2]["SOURCE"].startswith("Source S-"))


if __name__ == "__main__":
    unittest.main()
