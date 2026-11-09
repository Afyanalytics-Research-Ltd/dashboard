"""Discovery configuration: every rule is data here; classify.py and model/interpret.py hold the logic.

Names are matched on whole '_'-separated tokens of the upper-cased name (never substrings), except the few
*_SUFFIXES used for concatenated names such as CONSULTANTSVISITS. Each threshold notes its evidence.
Change a value here, run the tests, bump RULES_VERSION.
"""

RULES_VERSION = "2026-11-05.3"
DATABASE = "HOSPITALS"

# ------------------------------------------------------------------ schemas
SKIP_SCHEMAS = {"INFORMATION_SCHEMA", "ANALYTICS"}      # system schema; our own AFYA_* tables
RAW_SCHEMA_SUFFIX = "_RAW"                              # raw event stores: never candidates, used for lineage only
EXCLUDED_SCHEMAS = {                                    # decision D2 (2026-11-05): test, superseded, not in use
    "SILVERWOOD_TEST_CLEAN", "SPECIAL_CASES_TEST_CLEAN", "KISUMU_CLEAN",
    "AFYA_API_AUTH_CLEAN", "HANDWRITTEN_CLEAN", "HAND_WRITTEN_CLEAN", "SIAYA_CLEAN",
}
# a schema whose last meaningful token is one of these is a test/backup copy and is ignored
# (CSV_CONNECTION_TEST_CLEAN, V3_SYNC_CANARY, X_CLEAN_BAK); the word elsewhere (MIGORI_HIV_TEST_SITES_CLEAN) -> review
COPY_SCHEMA_TOKENS = {"TEST", "BACKUP", "BAK", "BKP", "COPY", "CANARY", "SANDBOX"}
HELD_SCHEMAS = {"XANALIFE_CLEAN"}                       # decision D3: facility-wide hold
DERIVED_SCHEMAS = {"STAGING", "REPORTING", "PUBLIC", "SILVERWOOD_EXTRACTION_MAPPED", "STAGING_V3_CORE_CLEAN"}
COVERED_SCHEMAS = {                                     # decision D7: exact schemas counted by an active source
    "ORTHOPEDIC_CLEAN": "HOSPITALS.STAGING.STG_VISITS",
    "ORTHOPEDIC_CLEAN_V2": "HOSPITALS.STAGING.STG_VISITS",
    "ORTHOPEDIC_CLEAN_V3": "HOSPITALS.STAGING.STG_VISITS",
}
# facility "family" = schema name with these trailing tokens removed (plus any trailing V<n>), e.g.
# KISUMU_V3_CLEAN -> KISUMU, EAGLE_NURSING_HOME_HISTORICAL_DATA -> EAGLE_NURSING_HOME, KAKAMEGA_RAW -> KAKAMEGA
FAMILY_TRAILING_TOKENS = {"CLEAN", "RAW", "HISTORICAL", "DATA", "LEGACY", "OLD", "ARCHIVE", "BACKUP", "TEST",
                          "BAK", "BKP", "COPY", "CANARY", "SANDBOX",
                          "EXPORT", "EXPORTS", "EXTRACT", "EXTRACTS", "EXTRACTION", "MAPPED", "DOCS", "CORE"}

# ------------------------------------------------------------------ person keys (column names)
PERSON_KEYS = {
    # tier 1: resolved / person-level identities
    "UNIQUE_PATIENT_ID": 1, "PERSON_ID": 1, "PATIENT_UUID": 1, "NUPI": 1,
    # tier 2: local patient numbers (proposable)
    "PATIENT_ID": 2, "PATIENT": 2, "PATIENTID": 2, "PATIENT_NO": 2, "PATIENT_NUMBER": 2, "PATIENTNUMBER": 2,
    "PATIENT_ID_HASH": 2, "REGID": 2, "REG_NO": 2, "REGISTRATION_NO": 2, "REGISTRATION_NUMBER": 2, "OPNO": 2,
    "OPD_NO": 2, "PNO": 2, "MRN": 2, "HOSPITAL_NO": 2, "CLIENT_ID": 2, "UPI": 2, "PATIENT_FILE_NO": 2,
    "CLINIC_PATIENT_ID": 2, "OP_NO": 2, "OP_NUMBER": 2,
    # tier 3: ambiguous until a person confirms (never proposed). MEMBERNO may be an insurance number (Eagle);
    # PATIENT_KEY is also an MPI key; COMPOSITE_PATIENT_ID crosses systems; SK_* may change per version.
    "MEMBERNO": 3, "MEMBER_NO": 3, "MEMBER_ID": 3, "MEMBERNUMBER": 3, "PATIENT_KEY": 3, "COMPOSITE_PATIENT_ID": 3,
    "NUPI_HASH": 3, "CCC_NUMBER_HASH": 3, "SK_PATIENT_ID": 3, "PATIENTSID": 3,
    # programme / admission numbers: person-level within a programme or stay, so a person confirms
    "CCC_NUMBER": 3, "CCC_NO": 3, "IP_NO": 3, "ANC_NO": 3,
}
KEY_TYPES = {"TEXT", "NUMBER", "FLOAT"}
# other names get tier 3 if they contain one of these tokens and end in an id-like token ...
KEY_VARIANT_TOKENS = {"PATIENT", "CLIENT", "PERSON"}
KEY_VARIANT_LAST = {"ID", "NO", "NUMBER", "UUID", "KEY"}
# ... unless they also contain an entity token (real look-alikes seen in INFORMATION_SCHEMA, 2026-11-04:
# PATIENT_DEPOSIT_ID, PATIENT_BILL_ITEM_ID, PATIENT_VISIT_NUMBER, PATIENT_BED_NO, RESPONSIBLE_PERSON_ID ...)
NOT_PERSON_KEY_TOKENS = {"BILL", "ITEM", "DEPOSIT", "WITHDRAWAL", "DEPENDANT", "ACCOUNT", "VISIT", "MOBILE",
                         "TELEPHONE", "PHONE", "NAME", "AGE", "KRA", "PIN", "SCAN", "SERVICE", "INVOICE", "EMPLOYEE",
                         "USER", "BED", "RECEIPT", "GUARDIAN", "KIN", "POLICY", "INSURANCE", "GROUP", "RESPONSIBLE",
                         "NURSE", "PREVIOUS", "PRINCIPAL", "NEW", "TSC", "NEMIS", "ALT", "CHIEF", "SYSTEM", "CHART"}

# ------------------------------------------------------------------ object roles (object names)
# The last token decides the role ("head noun"), after dropping ROLE_MODIFIER_TOKENS and V<n>/number tokens:
# EVALUATION_VISIT_CONSULTANTS -> satellite, INPATIENT_ADMISSION_TYPES -> other (a lookup), NANDI_OUTPATIENT_WIDE
# -> encounter. Only an encounter can be proposed; satellites are ignored.
ROLE_TOKENS = {
    "encounter": {"VISIT", "VISITS", "ENCOUNTER", "ENCOUNTERS", "ATTENDANCE", "ATTENDANCES", "ADMISSION",
                  "ADMISSIONS", "IPVISIT", "OUTPATIENT", "OUTPATIENTS", "CONSULTATION", "CONSULTATIONS", "OPD", "IPD"},
    "scheduling": {"APPOINTMENT", "APPOINTMENTS", "BOOKING", "BOOKINGS", "SCHEDULE", "SCHEDULES", "QUEUE"},
    "satellite": {"FINANCE", "INVOICE", "INVOICES", "PAYMENT", "PAYMENTS", "DEPOSIT", "DEPOSITS", "WAIVER",
                  "WAIVERS", "COPAY", "INVENTORY", "STOCK", "SALES", "NOK", "KIN", "INSURANCE", "SCHEME", "SCHEMES",
                  "DOCUMENT", "DOCUMENTS", "NOTE", "NOTES", "DEPENDANT", "DEPENDANTS", "VITAL", "VITALS", "ALLERGY",
                  "ALLERGIES", "DIAGNOSIS", "PRESCRIPTION", "PRESCRIPTIONS", "REQUEST", "REQUESTS", "BILL", "BILLS",
                  "CONSENT", "CONSENTS", "RECEIPT", "RECEIPTS", "TRANSACTION", "TRANSACTIONS", "CREDIT", "DEBIT",
                  "LAB", "LABS", "EXAMINATION", "HISTORY", "REFRACTION", "REVIEWS", "CONSULTANT", "CONSULTANTS",
                  "ORDER", "ORDERS", "PROCEDURE", "PROCEDURES", "IMAGING", "PHARMACY", "CHARGE", "CHARGES",
                  "INVESTIGATION", "INVESTIGATIONS", "SAMPLE", "SAMPLES", "RESULTS", "WITHDRAWALS", "ACCOUNTS"},
    "register": {"PATIENTS", "REGISTRATION", "REGISTRY", "REGISTER", "CLIENTS", "LINELIST"},
}
# decision D6 (2026-11-05): admissions add no patients beyond visits at counted facilities (aggregate check:
# 0 admitted patients without a visit at KSH, Tenri, Lodwar, Wema, Kakamega), so they are not separate sources
ADMISSION_TOKENS = {"ADMISSION", "ADMISSIONS", "IPVISIT", "IPD"}
ROLE_MODIFIER_TOKENS = {"WIDE", "COMBINED", "ALL", "FULL", "FINAL", "MERGED", "V"}
ENCOUNTER_SUFFIXES = ("VISIT", "VISITS", "VISITDETAILS")   # CONSULTANTSVISITS, LIMPATIENTVISITDETAILS
REGISTER_SUFFIXES = ("PATIENTS",)                          # LIMPATIENTS
COPY_TOKENS = {"TEST", "BACKUP", "BAK", "COPY", "TMP", "TEMP", "CANARY", "SANDBOX"}   # shown for review, never profiled
DATED_COPY_SUFFIX = r"__\d{8}$"                            # SILVERWOOD_TEST_CLEAN_BACKUP.VISITS__20261028
IDENTITY_TOKENS = {"MPI"}                                  # SMARTHEALTH_V3.MPI_CLUSTERS
IDENTITY_LAST_TOKENS = {"CLUSTERS"}

# ------------------------------------------------------------------ grain columns (exact names, first match wins)
RECORD_KEYS = ["VISIT_ID", "VISIT_NO", "VISIT_NUMBER", "VISIT_UUID", "ENCOUNTER_ID", "ENCOUNTER_NO", "ENCOUNTER_UUID",
               "APTID", "PRESCID", "PAY_VISIT_ID", "VISITID", "ADMISSION_ID", "ID"]
DELETED_COLUMNS = ["DELETED_AT", "IS_DELETED", "_IS_DELETED", "ISDELETED", "DELETED", "_DELETED_AT_", "_DELETED_AT",
                   "VOIDED", "ARCHIVED_AT"]
INVERTED_DELETED_COLUMNS = {"IS_ACTIVE"}       # true = live: must never be configured as DELETED_COLUMN
VERSION_COLUMNS = ["UPDATED_AT", "LAST_MODIFIED_ON", "_UPDATED_AT_", "DATEMODIFIED"]
RECEIVED_COLUMNS = ["_INGESTED_AT", "INGESTED_AT", "_RAW_INGESTED_AT", "_LOADED_AT", "LOADED_AT", "_EXTRACTED_AT"]
FILE_COLUMNS = ["FILENAME", "FILE_NAME", "SOURCE_FILE", "FILE_ID", "DOCUMENT_ID", "DOCUMENT_NO"]   # not FILE_NO
HOUSEHOLD_COLUMNS = ["HOUSEHOLD_ID", "HOUSEHOLD_NO", "HH_ID", "FAMILY_ID", "FAMILY_NO"]
DATE_TYPES = ("DATE", "TIMESTAMP_NTZ", "TIMESTAMP_LTZ", "TIMESTAMP_TZ")
NOT_VISIT_DATE_TOKENS = {"BIRTH", "DOB", "DELETED", "UPDATED", "MODIFIED", "INGESTED", "LOADED", "EXTRACTED"}

# the counting refresh's own deletion test (snowflake_patient_counts.sql): any other value means deleted
REFRESH_NOT_DELETED_VALUES = ("", "null", "NULL", "None", "none", "false", "FALSE", "0")
# identifiers the counting refresh accepts unquoted (snowflake_patient_counts.sql identifier checks)
REFRESH_IDENTIFIER = r"^[A-Z_][A-Z0-9_$]*$"
# key values treated as missing when profiling (sentinels seen in EHR extracts)
MISSING_KEY_VALUES = ("", "NULL", "NONE", "N/A", "NA", "-", "0", "-1", "UNKNOWN", "99999", "999999", "000000")

# ------------------------------------------------------------------ thresholds (any hit -> review)
SMALL_TABLE_ROWS = 500           # pilots/fixtures are 1-30 rows; too small to judge ratios
MISSING_KEY_SHARE = 0.05         # same 5% the refresh uses for its 'review' check
ONE_ROW_PER_KEY_SHARE = 0.90     # Siaya camp ~100%, SPH walk-ins ~100% within class; active EHR sources well below
KEY_EQUALS_RECORD_SHARE = 0.01   # patient ID literally equal to the visit ID (STG fallback); never coincidental at scale
DELETED_SHARE = 0.20             # active sources: Tenri 0.5%, Kakamega 0.08%
DUPLICATE_RECORD_SHARE = 0.01    # repeated (key, record) rows with no version column to collapse them
KEY_VARIANT_SHARE = 0.01         # '123' vs '123.0' vs 'p1'/'P1' for the same normalised key
MANY_RECORDS_PER_PERSON = 50     # mean records per key: a status / insurer / sci-notation column, not a person
RECORD_KEY_SHARED_SHARE = 0.01   # record IDs that span several persons (per-patient visit sequence, appended extracts)
DOMINANT_KEY_SHARE = 0.01        # one key holding >1% of rows ...
DOMINANT_KEY_VS_MEAN = 20        # ... and >20x the mean rows per key (an unlisted placeholder)
FINGERPRINT_MIN_ROWS = 100       # equal ROW_COUNT to an active source only counts above this size

# ------------------------------------------------------------------ run limits
MAX_PROFILES_PER_RUN = 10
MAX_RUN_SECONDS = 900
STATEMENT_TIMEOUT_SECONDS = 300  # session-level: applies to each profiling query separately
VIEW_PROFILE_MAX_AGE_DAYS = 28   # views have no reliable change marker; re-profile at most this often
