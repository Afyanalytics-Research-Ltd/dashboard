"""
build_ingest.py  —  Xana Snowflake Pipeline
══════════════════════════════════════════════════════════════════════════════
Step 1: Pull data from Snowflake EVENTS_RAW → MySQL xana_raw.

WHAT THIS DOES
──────────────
For each source schema in SNOWFLAKE_SOURCES and each table in TABLE_SPECS:

  1. On first run (or --full): DROP + CREATE target table, load all rows.
  2. On incremental runs: fetch only rows with updated_at > watermark,
     INSERT ... ON DUPLICATE KEY UPDATE into the existing table.
  3. After all raw tables are loaded, build evaluation_doctor_notes_merged —
     a pre-joined flat table of doctor_notes + ICD-10 hierarchy. This is
     what build_staging.py reads for NLP cleaning and ICD-10 coding.

RAW SCHEMA LAYOUT
──────────────────
All tables land in MySQL xana_raw with a source_schema column:

    xana_raw.evaluation_visits          → rows from all source schemas
    xana_raw.reception_patients         → ...
    xana_raw.evaluation_doctor_notes    → ...
    xana_raw.evaluation_icd10_notes     → ...
    xana_raw.evaluation_icd10_variations→ ...
    xana_raw.evaluation_icd10_types     → ...
    xana_raw.evaluation_icd10_subcategories → ...
    xana_raw.evaluation_icd10_categories    → ...
    xana_raw.evaluation_prescriptions   → ...
    xana_raw.finance_invoices           → ...
    xana_raw.evaluation_doctor_notes_merged  ← derived, built last

SURROGATE KEY STRATEGY
──────────────────────
Raw IDs are source-local. The surrogate key is:
    sk_visit_id    = "{source_schema}|{id}"   e.g. "tenri|42"
    sk_patient_id  = "{source_schema}|{id}"
These are computed in build_staging.py, not here.
The raw tables just carry source_schema + id.

INCREMENTAL BEHAVIOUR
──────────────────────
Watermark = MAX(updated_at) in the target MySQL table per source_schema.
Snowflake query adds: AND TRY_CAST(updated_at AS TIMESTAMP) > '{watermark}'
First run for a source_schema → watermark defaults to DATA_CUTOFF_DATE.

USAGE
──────
    python build_ingest.py                    # incremental all schemas
    python build_ingest.py --full             # full rebuild all schemas
    python build_ingest.py --schema tenri     # one schema only
    python build_ingest.py --table reception_patients  # one table only
    python build_ingest.py --totp 123456      # pass MFA code
══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import logging
import time
from typing import Optional

from sqlalchemy import text

from config import MYSQL_DATABASE, SNOWFLAKE_SOURCES, DATA_CUTOFF_DATE, raw_schema, fq, sf_events_raw
from db import build_mysql_engine, build_snowflake_conn

log = logging.getLogger(__name__)

# ── Table specs ───────────────────────────────────────────────────────────────
# Each spec defines one raw table. The same table receives rows from ALL
# source schemas — source_schema column distinguishes them.
#
# Fields:
#   sf_source_table   — WHERE source_table = '...' in Snowflake EVENTS_RAW
#   mysql_table       — target table name in xana_raw
#   pk                — primary key column (used in ON DUPLICATE KEY UPDATE)
#   watermark         — column for incremental loading
#   ddl               — MySQL CREATE TABLE DDL body
#   sf_select         — Snowflake SELECT body template
#                       {schema}    → replaced with source schema name
#                       {wm_filter} → replaced with watermark WHERE clause

TABLE_SPECS = [

    # ── Patients ──────────────────────────────────────────────────────────────
    {
        "sf_source_table": "reception_patients",
        "mysql_table":     "reception_patients",
        "pk":              "id",
        "watermark":       "updated_at",
        "ddl": """(
            id                  INT           NOT NULL,
            source_schema       VARCHAR(50)   NOT NULL,
            patient_no          INT,
            system_id           VARCHAR(100),
            user_id             INT,
            employee_id         INT,
            inpatient_no        VARCHAR(50),
            first_name          VARCHAR(100),
            middle_name         VARCHAR(100),
            last_name           VARCHAR(100),
            dob                 DATE,
            age_number          VARCHAR(20),
            age_in              VARCHAR(20),
            age_friendly        VARCHAR(50),
            sex                 VARCHAR(20),
            mobile              VARCHAR(50),
            id_no               VARCHAR(50),
            email               VARCHAR(200),
            address             VARCHAR(500),
            town                VARCHAR(100),
            nationality         VARCHAR(100),
            resident_county     VARCHAR(100),
            resident_sub_county VARCHAR(100),
            resident_village    VARCHAR(100),
            occupation          VARCHAR(200),
            marital_status      VARCHAR(50),
            ethnicity           VARCHAR(100),
            screening           INT,
            other_details       TEXT,
            registered_by       INT,
            source              VARCHAR(100),
            clinic_id           INT,
            status              INT,
            is_staff            INT,
            deceased            INT,
            eligible_for_points INT,
            schemes             TEXT,
            created_at          DATETIME,
            updated_at          DATETIME,
            deleted_at          DATETIME,
            PRIMARY KEY (id, source_schema),
            INDEX idx_updated       (updated_at),
            INDEX idx_source_schema (source_schema),
            INDEX idx_clinic        (clinic_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
        "sf_select": """
            SELECT DISTINCT
                f.value:id::INT                                                 AS id,
                '{schema}'                                                      AS source_schema,
                f.value:patient_no::INT                                         AS patient_no,
                f.value:system_id::STRING                                       AS system_id,
                f.value:user_id::INT                                            AS user_id,
                f.value:employee_id::INT                                        AS employee_id,
                f.value:inpatient_no::STRING                                    AS inpatient_no,
                f.value:first_name::STRING                                      AS first_name,
                f.value:middle_name::STRING                                     AS middle_name,
                f.value:last_name::STRING                                       AS last_name,
                CASE WHEN TRY_CAST(f.value:dob::STRING AS DATE) < '1900-01-01'
                     THEN NULL ELSE TRY_CAST(f.value:dob::STRING AS DATE) END  AS dob,
                f.value:age_number::STRING                                      AS age_number,
                f.value:age_in::STRING                                          AS age_in,
                f.value:age_friendly::STRING                                    AS age_friendly,
                f.value:sex::STRING                                             AS sex,
                f.value:mobile::STRING                                          AS mobile,
                f.value:id_no::STRING                                           AS id_no,
                f.value:email::STRING                                           AS email,
                f.value:address::STRING                                         AS address,
                f.value:town::STRING                                            AS town,
                f.value:nationality::STRING                                     AS nationality,
                f.value:resident_county::STRING                                 AS resident_county,
                f.value:resident_sub_county::STRING                             AS resident_sub_county,
                f.value:resident_village::STRING                                AS resident_village,
                f.value:occupation::STRING                                      AS occupation,
                f.value:marital_status::STRING                                  AS marital_status,
                f.value:ethnicity::STRING                                       AS ethnicity,
                f.value:screening::INT                                          AS screening,
                f.value:other_details::STRING                                   AS other_details,
                f.value:registered_by::INT                                      AS registered_by,
                f.value:source::STRING                                          AS source,
                f.value:clinic_id::INT                                          AS clinic_id,
                f.value:status::INT                                             AS status,
                f.value:is_staff::INT                                           AS is_staff,
                f.value:deceased::INT                                           AS deceased,
                f.value:eligible_for_points::INT                                AS eligible_for_points,
                f.value:schemes::STRING                                         AS schemes,
                TRY_CAST(f.value:created_at::STRING AS TIMESTAMP)              AS created_at,
                TRY_CAST(f.value:updated_at::STRING AS TIMESTAMP)              AS updated_at,
                TRY_CAST(f.value:deleted_at::STRING AS TIMESTAMP)              AS deleted_at
            FROM {events_raw},
            LATERAL FLATTEN(input => payload) f
            WHERE source_table = 'reception_patients'
              AND f.value:id::INT IS NOT NULL
              {wm_filter}
        """,
    },

    # ── Visits ────────────────────────────────────────────────────────────────
    {
        "sf_source_table": "evaluation_visits",
        "mysql_table":     "evaluation_visits",
        "pk":              "id",
        "watermark":       "updated_at",
        "ddl": """(
            id                  INT           NOT NULL,
            source_schema       VARCHAR(50)   NOT NULL,
            unique_id           VARCHAR(100),
            patient             INT,
            user                INT,
            staff_id            INT,
            clinic              INT,
            scheme              INT,
            corporate_id        INT,
            region_id           INT,
            payment_mode        VARCHAR(100),
            type_of_visit       VARCHAR(100),
            purpose             VARCHAR(500),
            status              VARCHAR(50),
            color_code          VARCHAR(20),
            referring_doctor    VARCHAR(200),
            admission_request   INT,
            inpatient           INT,
            theatre             INT,
            in_morgue           INT,
            external_order      VARCHAR(200),
            marked_as_em_by     INT,
            next_appointment    DATETIME,
            created_at          DATETIME,
            updated_at          DATETIME,
            deleted_at          DATETIME,
            PRIMARY KEY (id, source_schema),
            INDEX idx_updated       (updated_at),
            INDEX idx_source_schema (source_schema),
            INDEX idx_patient       (patient),
            INDEX idx_clinic        (clinic),
            INDEX idx_created       (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
        "sf_select": """
            SELECT DISTINCT
                f.value:id::INT                                             AS id,
                '{schema}'                                                  AS source_schema,
                f.value:unique_id::STRING                                   AS unique_id,
                f.value:patient::INT                                        AS patient,
                f.value:user::INT                                           AS user,
                f.value:staff_id::INT                                       AS staff_id,
                f.value:clinic::INT                                         AS clinic,
                f.value:scheme::INT                                         AS scheme,
                f.value:corporate_id::INT                                   AS corporate_id,
                f.value:region_id::INT                                      AS region_id,
                f.value:payment_mode::STRING                                AS payment_mode,
                f.value:type_of_visit::STRING                               AS type_of_visit,
                f.value:purpose::STRING                                     AS purpose,
                f.value:status::STRING                                      AS status,
                f.value:color_code::STRING                                  AS color_code,
                f.value:referring_doctor::STRING                            AS referring_doctor,
                f.value:admission_request::INT                              AS admission_request,
                f.value:inpatient::INT                                      AS inpatient,
                f.value:theatre::INT                                        AS theatre,
                f.value:in_morgue::INT                                      AS in_morgue,
                f.value:external_order::STRING                              AS external_order,
                f.value:marked_as_em_by::INT                                AS marked_as_em_by,
                TRY_CAST(f.value:next_appointment::STRING AS TIMESTAMP)     AS next_appointment,
                TRY_CAST(f.value:created_at::STRING AS TIMESTAMP)           AS created_at,
                TRY_CAST(f.value:updated_at::STRING AS TIMESTAMP)           AS updated_at,
                TRY_CAST(f.value:deleted_at::STRING AS TIMESTAMP)           AS deleted_at
            FROM {events_raw},
            LATERAL FLATTEN(input => payload) f
            WHERE source_table = 'evaluation_visits'
              AND f.value:id::INT IS NOT NULL
              {wm_filter}
        """,
    },

    # ── Doctor notes ──────────────────────────────────────────────────────────
    {
        "sf_source_table": "evaluation_doctor_notes",
        "mysql_table":     "evaluation_doctor_notes",
        "pk":              "id",
        "watermark":       "updated_at",
        "ddl": """(
            id                                  INT           NOT NULL,
            source_schema                       VARCHAR(50)   NOT NULL,
            visit                               INT,
            user                                INT,
            presenting_complaints               TEXT,
            chief_complaints                    TEXT,
            examination                         TEXT,
            systems_review                      TEXT,
            diagnosis                           TEXT,
            treatment_plan                      TEXT,
            treatment_or_prescription           TEXT,
            investigations                      TEXT,
            next_steps                          TEXT,
            remarks                             TEXT,
            past_medical_history                TEXT,
            duration_of_current_illness         VARCHAR(200),
            danger_signs                        TEXT,
            malaria                             TEXT,
            TB_screening                        TEXT,
            IMNCI_classifications_or_diagnosis  TEXT,
            mohDiagnosis                        TEXT,
            d1  VARCHAR(20), d2  VARCHAR(20), d3  VARCHAR(20), d4  VARCHAR(20),
            d5  VARCHAR(20), d6  VARCHAR(20), d7  VARCHAR(20), d8  VARCHAR(20),
            next_visit_date                     DATE,
            created_at                          DATETIME,
            updated_at                          DATETIME,
            PRIMARY KEY (id, source_schema),
            INDEX idx_visit         (visit),
            INDEX idx_source_schema (source_schema),
            INDEX idx_updated       (updated_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
        "sf_select": """
            SELECT DISTINCT
                f.value:id::INT                                         AS id,
                '{schema}'                                              AS source_schema,
                f.value:visit::INT                                      AS visit,
                f.value:user::INT                                       AS user,
                f.value:presenting_complaints::STRING                   AS presenting_complaints,
                f.value:chief_complaints::STRING                        AS chief_complaints,
                f.value:examination::STRING                             AS examination,
                f.value:systems_review::STRING                          AS systems_review,
                f.value:diagnosis::STRING                               AS diagnosis,
                f.value:treatment_plan::STRING                          AS treatment_plan,
                f.value:treatment_or_prescription::STRING               AS treatment_or_prescription,
                f.value:investigations::STRING                          AS investigations,
                f.value:next_steps::STRING                              AS next_steps,
                f.value:remarks::STRING                                 AS remarks,
                f.value:past_medical_history::STRING                    AS past_medical_history,
                f.value:duration_of_current_illness::STRING             AS duration_of_current_illness,
                f.value:danger_signs::STRING                            AS danger_signs,
                f.value:malaria::STRING                                 AS malaria,
                f.value:TB_screening::STRING                            AS TB_screening,
                f.value:IMNCI_classifications_or_diagnosis::STRING      AS IMNCI_classifications_or_diagnosis,
                f.value:mohDiagnosis::STRING                            AS mohDiagnosis,
                f.value:d1::STRING AS d1, f.value:d2::STRING AS d2,
                f.value:d3::STRING AS d3, f.value:d4::STRING AS d4,
                f.value:d5::STRING AS d5, f.value:d6::STRING AS d6,
                f.value:d7::STRING AS d7, f.value:d8::STRING AS d8,
                CASE WHEN TRY_CAST(f.value:next_visit_date::STRING AS DATE) < '1900-01-01'
                     THEN NULL ELSE TRY_CAST(f.value:next_visit_date::STRING AS DATE)
                END                                                     AS next_visit_date,
                TRY_CAST(f.value:created_at::STRING AS TIMESTAMP)       AS created_at,
                TRY_CAST(f.value:updated_at::STRING AS TIMESTAMP)       AS updated_at
            FROM {events_raw},
            LATERAL FLATTEN(input => payload) f
            WHERE source_table = 'evaluation_doctor_notes'
              AND f.value:id::INT IS NOT NULL
              {wm_filter}
        """,
    },

    # ── ICD-10 notes (visit → code link) ──────────────────────────────────────
    {
        "sf_source_table": "evaluation_icd10_notes",
        "mysql_table":     "evaluation_icd10_notes",
        "pk":              "id",
        "watermark":       "updated_at",
        "ddl": """(
            id              INT           NOT NULL,
            source_schema   VARCHAR(50)   NOT NULL,
            visit_id        INT,
            variation_id    INT,
            code            VARCHAR(20),
            created_at      DATETIME,
            updated_at      DATETIME,
            PRIMARY KEY (id, source_schema),
            INDEX idx_visit         (visit_id),
            INDEX idx_code          (code),
            INDEX idx_source_schema (source_schema),
            INDEX idx_updated       (updated_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
        "sf_select": """
            SELECT DISTINCT
                f.value:id::INT                                         AS id,
                '{schema}'                                              AS source_schema,
                f.value:visit_id::INT                                   AS visit_id,
                f.value:variation_id::INT                               AS variation_id,
                f.value:code::STRING                                    AS code,
                TRY_CAST(f.value:created_at::STRING AS TIMESTAMP)       AS created_at,
                TRY_CAST(f.value:updated_at::STRING AS TIMESTAMP)       AS updated_at
            FROM {events_raw},
            LATERAL FLATTEN(input => payload) f
            WHERE source_table = 'evaluation_icd10_notes'
              AND f.value:id::INT IS NOT NULL
              {wm_filter}
        """,
    },

    # ── ICD-10 variations ─────────────────────────────────────────────────────
    {
        "sf_source_table": "evaluation_icd10_variations",
        "mysql_table":     "evaluation_icd10_variations",
        "pk":              "id",
        "watermark":       "updated_at",
        "ddl": """(
            id              INT           NOT NULL,
            source_schema   VARCHAR(50)   NOT NULL,
            type_id         INT,
            code            VARCHAR(20),
            name            VARCHAR(500),
            link            VARCHAR(500),
            created_at      DATETIME,
            updated_at      DATETIME,
            PRIMARY KEY (id, source_schema),
            INDEX idx_code          (code),
            INDEX idx_type_id       (type_id),
            INDEX idx_source_schema (source_schema)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
        "sf_select": """
            SELECT DISTINCT
                f.value:id::INT                                         AS id,
                '{schema}'                                              AS source_schema,
                f.value:type_id::INT                                    AS type_id,
                f.value:code::STRING                                    AS code,
                f.value:name::STRING                                    AS name,
                f.value:link::STRING                                    AS link,
                TRY_CAST(f.value:created_at::STRING AS TIMESTAMP)       AS created_at,
                TRY_CAST(f.value:updated_at::STRING AS TIMESTAMP)       AS updated_at
            FROM {events_raw},
            LATERAL FLATTEN(input => payload) f
            WHERE source_table = 'evaluation_icd10_variations'
              AND f.value:id::INT IS NOT NULL
              {wm_filter}
        """,
    },

    # ── ICD-10 types ──────────────────────────────────────────────────────────
    {
        "sf_source_table": "evaluation_icd10_types",
        "mysql_table":     "evaluation_icd10_types",
        "pk":              "id",
        "watermark":       "updated_at",
        "ddl": """(
            id              INT           NOT NULL,
            source_schema   VARCHAR(50)   NOT NULL,
            subcategory_id  INT,
            code            VARCHAR(20),
            name            VARCHAR(500),
            link            VARCHAR(500),
            created_at      DATETIME,
            updated_at      DATETIME,
            PRIMARY KEY (id, source_schema),
            INDEX idx_code          (code),
            INDEX idx_source_schema (source_schema),
            INDEX idx_subcat        (subcategory_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
        "sf_select": """
            SELECT DISTINCT
                f.value:id::INT                                         AS id,
                '{schema}'                                              AS source_schema,
                f.value:subcategory_id::INT                             AS subcategory_id,
                f.value:code::STRING                                    AS code,
                f.value:name::STRING                                    AS name,
                f.value:link::STRING                                    AS link,
                TRY_CAST(f.value:created_at::STRING AS TIMESTAMP)       AS created_at,
                TRY_CAST(f.value:updated_at::STRING AS TIMESTAMP)       AS updated_at
            FROM {events_raw},
            LATERAL FLATTEN(input => payload) f
            WHERE source_table = 'evaluation_icd10_types'
              AND f.value:id::INT IS NOT NULL
              {wm_filter}
        """,
    },

    # ── ICD-10 subcategories ──────────────────────────────────────────────────
    {
        "sf_source_table": "evaluation_icd10_sub_categories",
        "mysql_table":     "evaluation_icd10_subcategories",
        "pk":              "id",
        "watermark":       "updated_at",
        "ddl": """(
            id              INT           NOT NULL,
            source_schema   VARCHAR(50)   NOT NULL,
            category_id     INT,
            code_from       VARCHAR(20),
            code_to         VARCHAR(20),
            name            VARCHAR(500),
            link            VARCHAR(500),
            created_at      DATETIME,
            updated_at      DATETIME,
            PRIMARY KEY (id, source_schema),
            INDEX idx_category      (category_id),
            INDEX idx_source_schema (source_schema)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
        "sf_select": """
            SELECT DISTINCT
                f.value:id::INT                                         AS id,
                '{schema}'                                              AS source_schema,
                f.value:category_id::INT                                AS category_id,
                f.value:code_from::STRING                               AS code_from,
                f.value:code_to::STRING                                 AS code_to,
                f.value:name::STRING                                    AS name,
                f.value:link::STRING                                    AS link,
                TRY_CAST(f.value:created_at::STRING AS TIMESTAMP)       AS created_at,
                TRY_CAST(f.value:updated_at::STRING AS TIMESTAMP)       AS updated_at
            FROM {events_raw},
            LATERAL FLATTEN(input => payload) f
            WHERE source_table = 'evaluation_icd10_sub_categories'
              AND f.value:id::INT IS NOT NULL
              {wm_filter}
        """,
    },

    # ── ICD-10 categories ─────────────────────────────────────────────────────
    {
        "sf_source_table": "evaluation_icd10_categories",
        "mysql_table":     "evaluation_icd10_categories",
        "pk":              "id",
        "watermark":       "updated_at",
        "ddl": """(
            id              INT           NOT NULL,
            source_schema   VARCHAR(50)   NOT NULL,
            code_from       VARCHAR(20),
            code_to         VARCHAR(20),
            name            VARCHAR(500),
            link            VARCHAR(500),
            created_at      DATETIME,
            updated_at      DATETIME,
            PRIMARY KEY (id, source_schema),
            INDEX idx_source_schema (source_schema)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
        "sf_select": """
            SELECT DISTINCT
                f.value:id::INT                                         AS id,
                '{schema}'                                              AS source_schema,
                f.value:code_from::STRING                               AS code_from,
                f.value:code_to::STRING                                 AS code_to,
                f.value:name::STRING                                    AS name,
                f.value:link::STRING                                    AS link,
                TRY_CAST(f.value:created_at::STRING AS TIMESTAMP)       AS created_at,
                TRY_CAST(f.value:updated_at::STRING AS TIMESTAMP)       AS updated_at
            FROM {events_raw},
            LATERAL FLATTEN(input => payload) f
            WHERE source_table = 'evaluation_icd10_categories'
              AND f.value:id::INT IS NOT NULL
              {wm_filter}
        """,
    },

    # ── Prescriptions ─────────────────────────────────────────────────────────
    {
        "sf_source_table": "evaluation_prescriptions",
        "mysql_table":     "evaluation_prescriptions",
        "pk":              "id",
        "watermark":       "updated_at",
        "ddl": """(
            id                      INT             NOT NULL,
            source_schema           VARCHAR(50)     NOT NULL,
            visit                   INT,
            user                    INT,
            drug_name               VARCHAR(500),
            drug_id                 INT,
            dosage                  VARCHAR(200),
            notes                   TEXT,
            quantity                INT,
            administered_quantity   INT,
            dispensed_quantity      INT,
            duration                INT,
            time_measure            INT,
            status                  INT,
            paid                    INT,
            dispensed               INT,
            stopped                 INT,
            canceled                INT,
            stop_reason             TEXT,
            admission_id            INT,
            facility_id             INT,
            store_id                INT,
            store_name              VARCHAR(200),
            prescribed_by           VARCHAR(200),
            price                   DECIMAL(18,2),
            created_at              DATETIME,
            updated_at              DATETIME,
            PRIMARY KEY (id, source_schema),
            INDEX idx_visit         (visit),
            INDEX idx_source_schema (source_schema),
            INDEX idx_updated       (updated_at),
            INDEX idx_drug          (drug_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
        "sf_select": """
            SELECT DISTINCT
                f.value:id::INT                                             AS id,
                '{schema}'                                                  AS source_schema,
                f.value:visit::INT                                          AS visit,
                f.value:user::INT                                           AS user,
                f.value:drug_name::STRING                                   AS drug_name,
                f.value:drug_id::INT                                        AS drug_id,
                f.value:dosage::STRING                                      AS dosage,
                f.value:notes::STRING                                       AS notes,
                f.value:quantity::INT                                       AS quantity,
                f.value:administered_quantity::INT                          AS administered_quantity,
                f.value:dispensed_quantity::INT                             AS dispensed_quantity,
                f.value:duration::INT                                       AS duration,
                f.value:time_measure::INT                                   AS time_measure,
                f.value:status::INT                                         AS status,
                f.value:paid::INT                                           AS paid,
                f.value:dispensed::INT                                      AS dispensed,
                f.value:stopped::INT                                        AS stopped,
                f.value:canceled::INT                                       AS canceled,
                f.value:stop_reason::STRING                                 AS stop_reason,
                f.value:admission_id::INT                                   AS admission_id,
                f.value:facility_id::INT                                    AS facility_id,
                f.value:store_id::INT                                       AS store_id,
                f.value:store_name::STRING                                  AS store_name,
                f.value:prescribed_by::STRING                               AS prescribed_by,
                f.value:price::DECIMAL(18,2)                                AS price,
                TRY_CAST(f.value:created_at::STRING AS TIMESTAMP)           AS created_at,
                TRY_CAST(f.value:updated_at::STRING AS TIMESTAMP)           AS updated_at
            FROM {events_raw},
            LATERAL FLATTEN(input => payload) f
            WHERE source_table = 'evaluation_prescriptions'
              AND f.value:id::INT IS NOT NULL
              {wm_filter}
        """,
    },

    # ── Finance invoices ──────────────────────────────────────────────────────
    {
        "sf_source_table": "finance_invoices",
        "mysql_table":     "finance_invoices",
        "pk":              "id",
        "watermark":       "updated_at",
        "ddl": """(
            id                  INT             NOT NULL,
            source_schema       VARCHAR(50)     NOT NULL,
            invoice_no          VARCHAR(100),
            invoice_no_prefix   VARCHAR(20),
            invoice_date        DATE,
            status              VARCHAR(50),
            source              VARCHAR(100),
            patient_id          INT,
            patient_no          VARCHAR(50),
            patient_name        VARCHAR(200),
            company_id          INT,
            scheme_id           INT,
            amount              DECIMAL(18,2),
            balance             DECIMAL(18,2),
            payable_amount      DECIMAL(18,2),
            paid                INT,
            for_cash            INT,
            user_id             INT,
            visit               INT,
            notes               TEXT,
            created_at          DATETIME,
            updated_at          DATETIME,
            deleted_at          DATETIME,
            PRIMARY KEY (id, source_schema),
            INDEX idx_visit         (visit),
            INDEX idx_patient       (patient_id),
            INDEX idx_source_schema (source_schema),
            INDEX idx_updated       (updated_at),
            INDEX idx_date          (invoice_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
        "sf_select": """
            SELECT DISTINCT
                f.value:id::INT                                                     AS id,
                '{schema}'                                                          AS source_schema,
                f.value:invoice_no::STRING                                          AS invoice_no,
                f.value:invoice_no_prefix::STRING                                   AS invoice_no_prefix,
                CASE WHEN TRY_CAST(f.value:invoice_date::STRING AS DATE) < '1900-01-01'
                     THEN NULL ELSE TRY_CAST(f.value:invoice_date::STRING AS DATE)
                END                                                                 AS invoice_date,
                f.value:status::STRING                                              AS status,
                f.value:source::STRING                                              AS source,
                f.value:patient_id::INT                                             AS patient_id,
                f.value:patient_no::STRING                                          AS patient_no,
                f.value:patient_name::STRING                                        AS patient_name,
                f.value:company_id::INT                                             AS company_id,
                f.value:scheme_id::INT                                              AS scheme_id,
                f.value:amount::DECIMAL(18,2)                                       AS amount,
                f.value:balance::DECIMAL(18,2)                                      AS balance,
                f.value:payable_amount::DECIMAL(18,2)                               AS payable_amount,
                f.value:paid::INT                                                   AS paid,
                f.value:for_cash::INT                                               AS for_cash,
                f.value:user_id::INT                                                AS user_id,
                f.value:visit::INT                                                  AS visit,
                f.value:notes::STRING                                               AS notes,
                TRY_CAST(f.value:created_at::STRING AS TIMESTAMP)                   AS created_at,
                TRY_CAST(f.value:updated_at::STRING AS TIMESTAMP)                   AS updated_at,
                TRY_CAST(f.value:deleted_at::STRING AS TIMESTAMP)                   AS deleted_at
            FROM {events_raw},
            LATERAL FLATTEN(input => payload) f
            WHERE source_table = 'finance_invoices'
              AND f.value:id::INT IS NOT NULL
              {wm_filter}
        """,
    },
]


# ── Merged doctor notes DDL ───────────────────────────────────────────────────
# Pre-joined flat table: doctor_notes + all 4 ICD-10 hierarchy tables.
# build_staging.py reads from this table instead of re-joining 5 tables
# every time. This is also the input to build_doctor_diagnosis_cleaned_v2.py.

_MERGED_DDL = """
CREATE TABLE IF NOT EXISTS `{staging}`.`evaluation_doctor_notes_merged` (
    -- Grain: one row per doctor_note row
    -- When a doctor_note has an icd10_note, the ICD-10 hierarchy is joined in.
    -- When no icd10_note exists, ICD-10 columns are NULL.

    -- Identity
    id                                  INT           NOT NULL,
    source_schema                       VARCHAR(50)   NOT NULL,
    visit                               INT,
    user                                INT,

    -- Clinical text (NLP input)
    presenting_complaints               TEXT,
    chief_complaints                    TEXT,
    examination                         TEXT,
    systems_review                      TEXT,
    diagnosis                           TEXT,
    treatment_plan                      TEXT,
    treatment_or_prescription           TEXT,
    investigations                      TEXT,
    remarks                             TEXT,
    past_medical_history                TEXT,
    duration_of_current_illness         VARCHAR(200),
    danger_signs                        TEXT,
    malaria                             TEXT,
    TB_screening                        TEXT,
    IMNCI_classifications_or_diagnosis  TEXT,
    mohDiagnosis                        TEXT,

    -- Raw ICD-10 code slots from doctor notes (d1..d8)
    d1 VARCHAR(20), d2 VARCHAR(20), d3 VARCHAR(20), d4 VARCHAR(20),
    d5 VARCHAR(20), d6 VARCHAR(20), d7 VARCHAR(20), d8 VARCHAR(20),

    -- From evaluation_icd10_notes (first matching code for this visit)
    note_id             INT,
    note_variation_id   INT,
    note_code           VARCHAR(20),

    -- ICD-10 hierarchy (variation → type → subcategory → category)
    icd10_variation_code            VARCHAR(20),
    icd10_variation_name            VARCHAR(500),
    icd10_type_code                 VARCHAR(20),
    icd10_type_name                 VARCHAR(500),
    icd10_subcategory_name          VARCHAR(500),
    icd10_subcategory_code_from     VARCHAR(20),
    icd10_subcategory_code_to       VARCHAR(20),
    icd10_category_name             VARCHAR(500),
    icd10_category_code_from        VARCHAR(20),
    icd10_category_code_to          VARCHAR(20),

    -- Timestamps
    next_visit_date     DATE,
    note_created_at     DATETIME,
    note_updated_at     DATETIME,

    PRIMARY KEY (id, source_schema),
    INDEX idx_visit         (visit),
    INDEX idx_source_schema (source_schema),
    INDEX idx_note_code     (note_code),
    INDEX idx_updated       (note_updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

_MERGED_INSERT = """
INSERT INTO `{staging}`.`evaluation_doctor_notes_merged` (
    id, source_schema, visit, user,
    presenting_complaints, chief_complaints, examination, systems_review,
    diagnosis, treatment_plan, treatment_or_prescription, investigations,
    remarks, past_medical_history, duration_of_current_illness,
    danger_signs, malaria, TB_screening, IMNCI_classifications_or_diagnosis,
    mohDiagnosis,
    d1, d2, d3, d4, d5, d6, d7, d8,
    note_id, note_variation_id, note_code,
    icd10_variation_code, icd10_variation_name,
    icd10_type_code, icd10_type_name,
    icd10_subcategory_name, icd10_subcategory_code_from, icd10_subcategory_code_to,
    icd10_category_name, icd10_category_code_from, icd10_category_code_to,
    next_visit_date, note_created_at, note_updated_at
)
SELECT
    dn.id,
    dn.source_schema,
    dn.visit,
    dn.user,
    dn.presenting_complaints,
    dn.chief_complaints,
    dn.examination,
    dn.systems_review,
    dn.diagnosis,
    dn.treatment_plan,
    dn.treatment_or_prescription,
    dn.investigations,
    dn.remarks,
    dn.past_medical_history,
    dn.duration_of_current_illness,
    dn.danger_signs,
    dn.malaria,
    dn.TB_screening,
    dn.IMNCI_classifications_or_diagnosis,
    dn.mohDiagnosis,
    dn.d1, dn.d2, dn.d3, dn.d4,
    dn.d5, dn.d6, dn.d7, dn.d8,
    -- First ICD-10 note for this visit (by earliest created_at)
    icn.id                          AS note_id,
    icn.variation_id                AS note_variation_id,
    COALESCE(icn.code, dn.d1)       AS note_code,   -- prefer structured code, fall back to d1
    icv.code                        AS icd10_variation_code,
    icv.name                        AS icd10_variation_name,
    ict.code                        AS icd10_type_code,
    ict.name                        AS icd10_type_name,
    ics.name                        AS icd10_subcategory_name,
    ics.code_from                   AS icd10_subcategory_code_from,
    ics.code_to                     AS icd10_subcategory_code_to,
    icc.name                        AS icd10_category_name,
    icc.code_from                   AS icd10_category_code_from,
    icc.code_to                     AS icd10_category_code_to,
    dn.next_visit_date,
    dn.created_at                   AS note_created_at,
    dn.updated_at                   AS note_updated_at
-- Reads from TENRI_RAW raw tables
FROM `{raw_dn}`.`evaluation_doctor_notes` dn
LEFT JOIN (
    SELECT visit_id, source_schema, id, variation_id, code,
           ROW_NUMBER() OVER (
               PARTITION BY visit_id, source_schema
               ORDER BY created_at ASC, id ASC
           ) AS rn
    FROM `{raw_icn}`.`evaluation_icd10_notes`
) icn
    ON  icn.visit_id      = dn.visit
    AND icn.source_schema  = dn.source_schema
    AND icn.rn             = 1
LEFT JOIN `{raw_icn}`.`evaluation_icd10_variations`    icv ON icv.id = icn.variation_id   AND icv.source_schema = dn.source_schema
LEFT JOIN `{raw_icn}`.`evaluation_icd10_types`         ict ON ict.id = icv.type_id         AND ict.source_schema = dn.source_schema
LEFT JOIN `{raw_icn}`.`evaluation_icd10_subcategories` ics ON ics.id = ict.subcategory_id  AND ics.source_schema = dn.source_schema
LEFT JOIN `{raw_icn}`.`evaluation_icd10_categories`    icc ON icc.id = ics.category_id     AND icc.source_schema = dn.source_schema
{watermark_filter}
ON DUPLICATE KEY UPDATE
    presenting_complaints           = VALUES(presenting_complaints),
    chief_complaints                = VALUES(chief_complaints),
    diagnosis                       = VALUES(diagnosis),
    treatment_plan                  = VALUES(treatment_plan),
    note_code                       = VALUES(note_code),
    icd10_variation_code            = VALUES(icd10_variation_code),
    icd10_variation_name            = VALUES(icd10_variation_name),
    icd10_type_code                 = VALUES(icd10_type_code),
    icd10_type_name                 = VALUES(icd10_type_name),
    icd10_subcategory_name          = VALUES(icd10_subcategory_name),
    icd10_category_name             = VALUES(icd10_category_name),
    note_updated_at                 = VALUES(note_updated_at)
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_schema(_engine, _schema: str) -> None:
    pass  # TENRI_RAW already exists — no CREATE SCHEMA permission needed


def _table_exists(engine, schema: str, table: str) -> bool:
    with engine.connect() as conn:
        n = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA=:db AND TABLE_NAME=:t"
        ), {"db": schema, "t": table}).scalar()
    return bool(n)


def _get_watermark(engine, mysql_schema: str, table: str, wm_col: str, source: str) -> str:
    """MAX(wm_col) for a specific source_schema, falls back to DATA_CUTOFF_DATE."""
    try:
        with engine.connect() as conn:
            val = conn.execute(text(
                f"SELECT MAX(`{wm_col}`) FROM `{mysql_schema}`.`{table}` "
                f"WHERE source_schema = :s"
            ), {"s": source}).scalar()
        if val is not None:
            return str(val)
    except Exception:
        pass
    return DATA_CUTOFF_DATE


def _fetch_sf(sf_conn, sql: str) -> tuple[list, list]:
    cs = sf_conn.cursor()
    try:
        cs.execute(sql)
        rows = cs.fetchall()
        cols  = [d[0].lower() for d in cs.description]
        return rows, cols
    finally:
        cs.close()


def _upsert(engine, mysql_schema: str, table: str, cols: list, rows: list, pk: str) -> int:
    if not rows:
        return 0
    col_list = ", ".join(f"`{c}`" for c in cols)
    ph       = ", ".join([f":{c}" for c in cols])
    updates  = ", ".join(
        f"`{c}` = VALUES(`{c}`)"
        for c in cols
        if c not in (pk, "source_schema", "id")
    )
    sql = (
        f"INSERT INTO `{mysql_schema}`.`{table}` ({col_list}) VALUES ({ph}) "
        f"ON DUPLICATE KEY UPDATE {updates}"
    )
    BATCH = 2000
    total = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), BATCH):
            conn.execute(text(sql), [dict(zip(cols, r)) for r in rows[i:i+BATCH]])
            total += len(rows[i:i+BATCH])
    return total


# ── Local MySQL fallback ──────────────────────────────────────────────────────

LOCAL_FALLBACK_SCHEMA = "tenri"   # local MySQL schema to copy from when table is empty


def _row_count(engine, schema: str, table: str, source: str) -> int:
    """Count rows in target table for a given source_schema."""
    try:
        with engine.connect() as conn:
            return conn.execute(text(
                f"SELECT COUNT(*) FROM `{schema}`.`{table}` WHERE source_schema = :s"
            ), {"s": source}).scalar() or 0
    except Exception:
        return 0


def _fallback_from_local(engine, target_schema: str, table: str, logical_name: str) -> None:
    """
    Copy rows from the local MySQL fallback schema into the (empty) target table.
    Handles the source_schema column: adds it as a literal if absent in the source.
    Skips silently if the local table does not exist.
    """
    # Check local table exists
    with engine.connect() as conn:
        exists = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = :s AND TABLE_NAME = :t"
        ), {"s": LOCAL_FALLBACK_SCHEMA, "t": table}).scalar()
    if not exists:
        log.warning("  FALLBACK  %-40s local table `%s`.`%s` not found — skipping",
                    table, LOCAL_FALLBACK_SCHEMA, table)
        return

    # Get column names from both sides
    with engine.connect() as conn:
        target_cols = [
            r[0] for r in conn.execute(text(
                f"SELECT COLUMN_NAME FROM information_schema.COLUMNS "
                f"WHERE TABLE_SCHEMA = :s AND TABLE_NAME = :t ORDER BY ORDINAL_POSITION"
            ), {"s": target_schema, "t": table}).fetchall()
        ]
        source_cols = [
            r[0] for r in conn.execute(text(
                f"SELECT COLUMN_NAME FROM information_schema.COLUMNS "
                f"WHERE TABLE_SCHEMA = :s AND TABLE_NAME = :t ORDER BY ORDINAL_POSITION"
            ), {"s": LOCAL_FALLBACK_SCHEMA, "t": table}).fetchall()
        ]

    if not target_cols or not source_cols:
        log.warning("  FALLBACK  %-40s could not read columns — skipping", table)
        return

    source_col_set = set(source_cols)
    select_parts = []
    for col in target_cols:
        if col == "source_schema":
            select_parts.append(f"'{logical_name}' AS source_schema")
        elif col in source_col_set:
            select_parts.append(f"`{col}`")
        else:
            select_parts.append(f"NULL AS `{col}`")

    select_clause = ", ".join(select_parts)
    insert_sql = (
        f"INSERT IGNORE INTO `{target_schema}`.`{table}` "
        f"SELECT {select_clause} FROM `{LOCAL_FALLBACK_SCHEMA}`.`{table}`"
    )

    with engine.begin() as conn:
        result = conn.execute(text(insert_sql))
    log.info("  FALLBACK  %-40s [%s] %d rows copied from local `%s`",
             table, logical_name, result.rowcount, LOCAL_FALLBACK_SCHEMA)


# ── Per-table ingest ──────────────────────────────────────────────────────────

def _ingest_table(sf_conn, engine, spec: dict, logical_name: str, events_raw: str, full: bool) -> None:
    mschema = raw_schema(logical_name)   # MySQL schema e.g. TENRI_RAW
    table   = spec["mysql_table"]
    pk      = spec["pk"]
    wm      = spec["watermark"]

    # Create table if needed
    if full or not _table_exists(engine, mschema, table):
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS `{mschema}`.`{table}`"))
            conn.execute(text(f"CREATE TABLE `{mschema}`.`{table}` {spec['ddl']}"))
        watermark_val = DATA_CUTOFF_DATE
        mode = "FULL"
    else:
        watermark_val = _get_watermark(engine, mschema, table, wm, logical_name)
        mode = "INCR"

    # Skip-if-exists: table already has data → switch to incremental safely.
    # This means first run on an already-populated schema won't re-load everything.
    if mode == "FULL":
        with engine.connect() as conn:
            existing = conn.execute(text(
                f"SELECT COUNT(*) FROM `{mschema}`.`{table}` "
                f"WHERE source_schema = :s LIMIT 1"
            ), {"s": logical_name}).scalar()
        if existing:
            watermark_val = _get_watermark(engine, mschema, table, wm, logical_name)
            mode = "INCR"
            log.info("  EXIST  %-45s [%s] has data — using incremental", table, logical_name)

    wm_filter = (
        f"AND TRY_CAST(f.value:{wm}::STRING AS TIMESTAMP) > '{watermark_val}'"
        if mode == "INCR" else ""
    )

    sf_sql = spec["sf_select"].format(
        schema=logical_name,
        events_raw=events_raw,
        wm_filter=wm_filter,
    )

    t0 = time.time()
    rows, cols = _fetch_sf(sf_conn, sf_sql)
    elapsed = time.time() - t0

    if not rows:
        log.info("  SKIP   %-45s [%s] no new rows since %s", table, logical_name, watermark_val)
        return

    n = _upsert(engine, mschema, table, cols, rows, pk)
    log.info("  %-5s  %-45s [%s] %d rows  (%.1fs)", mode, table, logical_name, n, elapsed)


# ── Merged notes ──────────────────────────────────────────────────────────────

def _build_merged_notes(engine, full: bool) -> None:
    """
    Build evaluation_doctor_notes_merged in STAGING schema.
    Joins doctor_notes + ICD-10 hierarchy from all source schemas into one flat table.
    Lives in STAGING not in a raw schema since it merges across all sources.
    """
    from config import STAGING_SCHEMA
    merged = "evaluation_doctor_notes_merged"
    if full or not _table_exists(engine, STAGING_SCHEMA, merged):
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS `{STAGING_SCHEMA}`.`{merged}`"))
            conn.execute(text(_MERGED_DDL.format(staging=STAGING_SCHEMA)))
        wm_filter = ""
        log.info("  BUILD  %-45s full rebuild", merged)
    else:
        with engine.connect() as conn:
            wm = conn.execute(text(
                f"SELECT MAX(note_updated_at) FROM `{STAGING_SCHEMA}`.`{merged}`"
            )).scalar()
        wm_val = str(wm) if wm else DATA_CUTOFF_DATE
        wm_filter = f"WHERE dn.updated_at > '{wm_val}'"
        log.info("  MERGE  %-45s incremental since %s", merged, wm_val)

    raw = raw_schema("tenri")
    insert_sql = _MERGED_INSERT.format(
        staging=STAGING_SCHEMA,
        raw_dn=raw,
        raw_icn=raw,
        watermark_filter=wm_filter,
    )
    with engine.begin() as conn:
        result = conn.execute(text(insert_sql))
    log.info("  MERGE  %-45s %d rows upserted", merged, result.rowcount)


# ── Public entry point ────────────────────────────────────────────────────────

def run_ingest(
    engine,
    full: bool = False,
    schema_filter: Optional[str] = None,
    table_filter:  Optional[str] = None,
    totp:          Optional[str] = None,
    sf_conn=None,
) -> None:
    """
    Main entry point called from run_pipeline.py Step 1.
    Pulls Snowflake EVENTS_RAW → MySQL xana_raw.
    sf_conn: pass an existing connection to reuse (won't be closed here).
    """
    log.info("=" * 60)
    log.info("Step 1: Snowflake ingest (%s)", "FULL" if full else "INCREMENTAL")
    log.info("=" * 60)

    # Ensure all raw schemas exist in MySQL
    for logical_name, sf_schema in SNOWFLAKE_SOURCES.items():
        _ensure_schema(engine, sf_schema)
    from config import STAGING_SCHEMA
    _ensure_schema(engine, STAGING_SCHEMA)

    _owns_conn = sf_conn is None
    if _owns_conn:
        sf_conn = build_snowflake_conn(totp)
    try:
        for logical_name, sf_schema in SNOWFLAKE_SOURCES.items():
            if schema_filter and logical_name != schema_filter:
                continue

            er = sf_events_raw(sf_schema)    # e.g. HOSPITALS.TENRI_RAW.EVENTS_RAW
            log.info("  Schema: %s  (%s)", logical_name, er)

            for spec in TABLE_SPECS:
                if table_filter and spec["mysql_table"] != table_filter:
                    continue
                try:
                    _ingest_table(sf_conn, engine, spec, logical_name, er, full)
                    mschema = raw_schema(logical_name)
                    if _row_count(engine, mschema, spec["mysql_table"], logical_name) == 0:
                        log.warning("  EMPTY  %-40s [%s] — falling back to local `%s`",
                                    spec["mysql_table"], logical_name, LOCAL_FALLBACK_SCHEMA)
                        _fallback_from_local(engine, mschema, spec["mysql_table"], logical_name)
                except Exception as exc:
                    log.error("  FAIL  %s [%s] %s", spec["mysql_table"], logical_name, exc)
                    raise

        # Merged notes rebuilt after all raw tables are current
        if not table_filter or table_filter == "evaluation_doctor_notes_merged":
            _build_merged_notes(engine, full)

    finally:
        if _owns_conn:
            sf_conn.close()

    log.info("=" * 60)
    log.info("Step 1 complete")
    log.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Xana: Snowflake → MySQL raw ingest")
    parser.add_argument("--full",   action="store_true", help="Full rebuild all tables")
    parser.add_argument("--schema", default=None,        help="Only this source schema")
    parser.add_argument("--table",  default=None,        help="Only this MySQL table")
    parser.add_argument("--totp",   default=None,        help="Snowflake MFA passcode")
    args = parser.parse_args()

    if not args.totp and not os.getenv("SF_PASSWORD"):
        args.totp = input("Snowflake MFA code (Enter to skip): ").strip() or None

    engine = build_mysql_engine()
    run_ingest(engine, full=args.full, schema_filter=args.schema,
               table_filter=args.table, totp=args.totp)