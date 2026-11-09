"""Shared naming and normalisation rules (classify.py, profile_sql.py, config.py)."""
import unittest

from patient_discovery import config as C
from patient_discovery.classify import copy_schema, family, key_tier, role
from patient_discovery.profile_sql import normalise_key, profile_sql

from .fixtures import VISIT_COLS, classified, obj


class Rules(unittest.TestCase):
    def test_key_names(self):
        self.assertEqual(key_tier("PATIENT_NO", "TEXT"), 2)
        self.assertEqual(key_tier("PATIENT_ID", "BOOLEAN"), None)
        for look_alike in ("PATIENT_DEPOSIT_ID", "PATIENT_VISIT_NUMBER", "PATIENT_BED_NO", "RESPONSIBLE_PERSON_ID",
                           "INPATIENT_ID", "PATIENT_MOBILE_HASH", "NEW_PATIENT_SERVICE_ID", "PATIENTVISITID"):
            self.assertIsNone(key_tier(look_alike, "TEXT"), look_alike)
        self.assertEqual(key_tier("PATIENT_ROLL_ID", "TEXT"), 3)

    def test_identifier_endings_are_words_not_letters(self):
        from patient_discovery.model.interpret import id_named, nominate
        for yes in ("PATIENT_ID", "PATIENTID", "ID", "PHONE_NO", "UHID", "VISIT_UUID", "REG_NUMBER", "_KEY"):
            self.assertTrue(id_named(yes), yes)
        for no in ("AMOUNT_PAID", "IS_VALID", "PREPAID", "VOID", "THYROID", "FLUID", "STATUS", "ID_TYPE"):
            self.assertFalse(id_named(no), no)
        cands, _ = nominate([("AMOUNT_PAID", "TEXT"), ("CLIENT_ID", "TEXT")], set())
        self.assertEqual([c.column for c in cands], ["CLIENT_ID"])

    def test_roles_use_whole_tokens(self):
        self.assertEqual(role("LABOUR_AND_DELIVERY")[0], "other")       # not LAB
        self.assertEqual(role("DISORDER_REGISTER")[0], "register")      # not ORDER
        self.assertEqual(role("EVALUATION_VISIT_CONSULTANTS")[0], "satellite")
        self.assertEqual(role("INPATIENT_ADMISSION_REQUESTS")[0], "satellite")
        self.assertEqual(role("NANDI_OUTPATIENT_WIDE")[0], "encounter")
        self.assertEqual(role("CHAMPION_VISITS")[0], "encounter")       # not MPI
        self.assertEqual(role("LAB_VISITS"), ("encounter", True))

    def test_families_and_copy_schemas(self):
        self.assertEqual(family("KISUMU_V3_CLEAN"), "KISUMU")
        self.assertEqual(family("EAGLE_NURSING_HOME_HISTORICAL_DATA"), "EAGLE_NURSING_HOME")
        self.assertEqual(family("ORTHOPEDIC_HOSPITAL_NAKURU_CLEAN"), "ORTHOPEDIC_HOSPITAL_NAKURU")
        self.assertEqual(family("STAGING"), "STAGING")
        self.assertTrue(copy_schema("CSV_CONNECTION_TEST_CLEAN"))
        self.assertFalse(copy_schema("MIGORI_HIV_TEST_SITES_CLEAN"))

    def test_key_normalisation(self):
        self.assertEqual(normalise_key("123", "TEXT"), normalise_key("123.0", "TEXT"))
        self.assertEqual(normalise_key(123, "NUMBER"), normalise_key("123", "TEXT"))
        self.assertEqual(normalise_key(" p1 ", "TEXT"), "P1")
        for missing in ("", " ", "null", "N/A", "0", "-1", "00", "UNKNOWN"):
            self.assertIsNone(normalise_key(missing, "TEXT"), missing)
        self.assertIsNone(normalise_key(0.0, "FLOAT"))

    def test_snowflake_profile_sql_shape(self):
        c = classified(obj("NEWFAC_CLEAN", "VISITS", VISIT_COLS, []))
        sql = profile_sql(c, "snowflake")
        self.assertIn('FROM "HOSPITALS"."NEWFAC_CLEAN"."VISITS"', sql)
        self.assertIn("TRY_TO_NUMBER(", sql)
        self.assertEqual(sql.count("FROM"), 2)                     # one read of the object
        self.assertNotRegex(sql, r"(?i)(insert|update|delete|merge|create|drop|sample)")

    def test_config_is_consistent(self):
        self.assertFalse(set(C.EXCLUDED_SCHEMAS) & set(C.HELD_SCHEMAS))
        self.assertFalse(set(C.COVERED_SCHEMAS) & set(C.EXCLUDED_SCHEMAS))
        for r, toks in C.ROLE_TOKENS.items():
            for other, toks2 in C.ROLE_TOKENS.items():
                if r < other:
                    self.assertFalse(toks & toks2, (r, other, toks & toks2))


if __name__ == "__main__":
    unittest.main()
