"""
build_staging.py  —  Xana Pipeline
══════════════════════════════════════════════════════════════════════════════
Step 2: Build unified staging tables in HOSPITALS.STAGING.

READS FROM
──────────
    HOSPITALS.TENRI_RAW.*     — raw tables from tenri source
    HOSPITALS.KAKAMEGA_RAW.*  — raw tables from kakamega source
    HOSPITALS.LODWAR_RAW.*    — raw tables from lodwar source
    HOSPITALS.KISUMU_RAW.*    — raw tables from kisumu source
    HOSPITALS.STAGING.evaluation_doctor_notes_merged  — pre-joined notes (all sources)

WRITES TO
──────────
    HOSPITALS.STAGING.stg_visits
    HOSPITALS.STAGING.stg_patients
    HOSPITALS.STAGING.stg_doctor_notes     ← reads from merged table
    HOSPITALS.STAGING.stg_icd10            ← reads from merged table
    HOSPITALS.STAGING.stg_prescriptions
    HOSPITALS.STAGING.stg_invoices

SURROGATE KEYS
──────────────
sk_visit_id    = CONCAT(source_schema, '|', id)   e.g. "tenri|42"
sk_patient_id  = CONCAT(source_schema, '|', id)
Computed inline — no separate bridge table needed.

MULTI-SOURCE UNION
───────────────────
For raw tables (patients, visits, prescriptions, invoices), the staging
tables UNION ALL across all source schemas. The source_schema column
distinguishes rows.

For doctor_notes and icd10, the merged table in STAGING already contains
rows from all sources, so no UNION is needed.
══════════════════════════════════════════════════════════════════════════════
"""

import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from config import (
    MYSQL_DATABASE, STAGING_SCHEMA, DATA_CUTOFF_DATE,
    SNOWFLAKE_SOURCES, raw_schema
)

log = logging.getLogger(__name__)

C = "utf8mb4_unicode_ci"


def _ensure_staging_schema(_engine: Engine) -> None:
    pass  # TENRI_RAW already exists — no CREATE DATABASE permission needed


def _table_exists(engine: Engine, table: str) -> bool:
    with engine.connect() as conn:
        n = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA=:db AND TABLE_NAME=:t"
        ), {"db": STAGING_SCHEMA, "t": table}).scalar()
    return bool(n)


def _get_watermark(engine: Engine, table: str, wm_col: str, source: str) -> str:
    try:
        with engine.connect() as conn:
            val = conn.execute(text(
                f"SELECT MAX(`{wm_col}`) FROM `{STAGING_SCHEMA}`.`{table}` "
                f"WHERE source_schema = :s"
            ), {"s": source}).scalar()
        if val is not None:
            return str(val)
    except Exception:
        pass
    return DATA_CUTOFF_DATE


def _full_rebuild(engine: Engine, table: str, ddl: str, union_sql: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS `{STAGING_SCHEMA}`.`{table}`"))
        conn.execute(text(f"CREATE TABLE `{STAGING_SCHEMA}`.`{table}` {ddl} AS\n{union_sql}"))
    log.info("  FULL   %-45s rebuilt", table)


def _incremental(engine: Engine, table: str, select_sql: str, wm_col: str, source: str) -> None:
    wm = _get_watermark(engine, table, wm_col, source)
    with engine.connect() as conn:
        cols = [r[0] for r in conn.execute(text(
            f"SHOW COLUMNS FROM `{STAGING_SCHEMA}`.`{table}`"
        )).fetchall()]
    col_list = ", ".join(f"`{c}`" for c in cols)
    updates = ", ".join(
        f"`{c}` = VALUES(`{c}`)"
        for c in cols
        if c not in ("sk_visit_id", "sk_patient_id", "sk_prescription_id",
                     "sk_invoice_id", "source_schema")
    )
    insert_sql = (
        f"INSERT INTO `{STAGING_SCHEMA}`.`{table}` ({col_list})\n"
        f"{select_sql}\n"
        f"ON DUPLICATE KEY UPDATE {updates}"
    )
    with engine.begin() as conn:
        result = conn.execute(text(insert_sql))
    n = result.rowcount
    if n:
        log.info("  INCR   %-45s [%s] %d rows", table, source, n)
    else:
        log.info("  SKIP   %-45s [%s] no new rows since %s", table, source, wm)


# ── stg_visits ────────────────────────────────────────────────────────────────

STG_VISITS_DDL = f"""(
    sk_visit_id      VARCHAR(100) NOT NULL,
    source_schema    VARCHAR(50)  NOT NULL,
    raw_id           INT          NOT NULL,
    raw_patient_id   INT,
    sk_patient_id    VARCHAR(100),
    facility_key     VARCHAR(100),
    unique_id        VARCHAR(100) COLLATE {C},
    payment_mode     VARCHAR(100) COLLATE {C},
    type_of_visit    VARCHAR(100) COLLATE {C},
    scheme           INT,
    region_id        INT,
    in_morgue        INT,
    visit_type       VARCHAR(20)  COLLATE {C},
    visit_start_date DATETIME,
    visit_updated_at DATETIME,
    deleted_at       DATETIME,
    PRIMARY KEY (sk_visit_id),
    INDEX idx_patient  (sk_patient_id),
    INDEX idx_facility (facility_key),
    INDEX idx_source   (source_schema),
    INDEX idx_date     (visit_start_date),
    INDEX idx_updated  (visit_updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE={C}"""


def _visits_select(src_schema: str, logical: str, wm_filter: str = "") -> str:
    return f"""
    SELECT
        CONCAT('{logical}', '|', src.id)        AS sk_visit_id,
        '{logical}'                              AS source_schema,
        src.id                                   AS raw_id,
        src.patient                              AS raw_patient_id,
        CONCAT('{logical}', '|', src.patient)    AS sk_patient_id,
        CONCAT('{logical}', '|', src.clinic)     AS facility_key,
        CONVERT(IFNULL(src.unique_id,'')    USING utf8mb4) COLLATE {C} AS unique_id,
        CONVERT(IFNULL(src.payment_mode,'') USING utf8mb4) COLLATE {C} AS payment_mode,
        CONVERT(IFNULL(src.type_of_visit,'') USING utf8mb4) COLLATE {C} AS type_of_visit,
        src.scheme,
        src.region_id,
        src.in_morgue,
        CASE WHEN src.inpatient = 1 THEN 'inpatient' ELSE 'outpatient' END AS visit_type,
        src.created_at   AS visit_start_date,
        src.updated_at   AS visit_updated_at,
        src.deleted_at
    FROM `{src_schema}`.`evaluation_visits` src
    WHERE src.clinic IS NOT NULL
      AND src.created_at >= '{DATA_CUTOFF_DATE}'
      {wm_filter}
    """


def build_stg_visits(engine: Engine, full: bool = False) -> None:
    table = "stg_visits"
    if full or not _table_exists(engine, table):
        union_parts = [
            _visits_select(raw_schema(lg), lg)
            for lg in SNOWFLAKE_SOURCES.keys()
        ]
        _full_rebuild(engine, table, STG_VISITS_DDL,
                      "\nUNION ALL\n".join(union_parts))
    else:
        for logical in SNOWFLAKE_SOURCES.keys():
            wm = _get_watermark(engine, table, "visit_updated_at", logical)
            wm_filter = f"AND src.updated_at > '{wm}'"
            _incremental(engine, table,
                         _visits_select(raw_schema(logical), logical, wm_filter),
                         "visit_updated_at", logical)


# ── stg_patients ──────────────────────────────────────────────────────────────

STG_PATIENTS_DDL = f"""(
    sk_patient_id      VARCHAR(100) NOT NULL,
    source_schema      VARCHAR(50)  NOT NULL,
    raw_id             INT          NOT NULL,
    patient_no         INT,
    inpatient_no       VARCHAR(50)  COLLATE {C},
    sex                VARCHAR(20)  COLLATE {C},
    dob                DATE,
    marital_status     VARCHAR(50)  COLLATE {C},
    resident_county    VARCHAR(100) COLLATE {C},
    resident_sub_county VARCHAR(100) COLLATE {C},
    status             INT,
    patient_created_at DATETIME,
    patient_updated_at DATETIME,
    PRIMARY KEY (sk_patient_id),
    INDEX idx_source  (source_schema),
    INDEX idx_sex     (sex),
    INDEX idx_updated (patient_updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE={C}"""


def _patients_select(src_schema: str, logical: str, wm_filter: str = "") -> str:
    return f"""
    SELECT
        CONCAT('{logical}', '|', src.id)    AS sk_patient_id,
        '{logical}'                          AS source_schema,
        src.id                               AS raw_id,
        src.patient_no,
        CONVERT(IFNULL(src.inpatient_no,'')    USING utf8mb4) COLLATE {C} AS inpatient_no,
        CONVERT(IFNULL(src.sex,'')             USING utf8mb4) COLLATE {C} AS sex,
        src.dob,
        CONVERT(IFNULL(src.marital_status,'')  USING utf8mb4) COLLATE {C} AS marital_status,
        CONVERT(IFNULL(src.resident_county,'') USING utf8mb4) COLLATE {C} AS resident_county,
        CONVERT(IFNULL(src.resident_sub_county,'') USING utf8mb4) COLLATE {C} AS resident_sub_county,
        src.status,
        src.created_at  AS patient_created_at,
        src.updated_at  AS patient_updated_at
    FROM `{src_schema}`.`reception_patients` src
    WHERE src.deleted_at IS NULL
      {wm_filter}
    """


def build_stg_patients(engine: Engine, full: bool = False) -> None:
    table = "stg_patients"
    if full or not _table_exists(engine, table):
        union_parts = [
            _patients_select(raw_schema(lg), lg)
            for lg in SNOWFLAKE_SOURCES.keys()
        ]
        _full_rebuild(engine, table, STG_PATIENTS_DDL,
                      "\nUNION ALL\n".join(union_parts))
    else:
        for logical in SNOWFLAKE_SOURCES.keys():
            wm = _get_watermark(engine, table, "patient_updated_at", logical)
            wm_filter = f"AND src.updated_at > '{wm}'"
            _incremental(engine, table,
                         _patients_select(raw_schema(logical), logical, wm_filter),
                         "patient_updated_at", logical)


# ── stg_doctor_notes ──────────────────────────────────────────────────────────
# Reads from STAGING.evaluation_doctor_notes_merged — already merged across sources.

STG_DOCTOR_NOTES_DDL = f"""(
    sk_visit_id           VARCHAR(100) NOT NULL,
    source_schema         VARCHAR(50)  NOT NULL,
    presenting_complaints TEXT         COLLATE {C},
    chief_complaints      TEXT         COLLATE {C},
    examination           TEXT         COLLATE {C},
    systems_review        TEXT         COLLATE {C},
    diagnosis             TEXT         COLLATE {C},
    treatment_plan        TEXT         COLLATE {C},
    investigations        TEXT         COLLATE {C},
    past_medical_history  TEXT         COLLATE {C},
    remarks               TEXT         COLLATE {C},
    note_created_at       DATETIME,
    note_updated_at       DATETIME,
    PRIMARY KEY (sk_visit_id, source_schema),
    INDEX idx_source  (source_schema),
    INDEX idx_updated (note_updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE={C}"""


def _doctor_notes_select(wm_filter: str = "") -> str:
    # One row per visit — latest note wins when a visit has multiple notes
    return f"""
    SELECT
        CONCAT(src.source_schema, '|', src.visit)  AS sk_visit_id,
        src.source_schema,
        CONVERT(IFNULL(src.presenting_complaints,'') USING utf8mb4) COLLATE {C} AS presenting_complaints,
        CONVERT(IFNULL(src.chief_complaints,'')      USING utf8mb4) COLLATE {C} AS chief_complaints,
        CONVERT(IFNULL(src.examination,'')           USING utf8mb4) COLLATE {C} AS examination,
        CONVERT(IFNULL(src.systems_review,'')        USING utf8mb4) COLLATE {C} AS systems_review,
        CONVERT(IFNULL(src.diagnosis,'')             USING utf8mb4) COLLATE {C} AS diagnosis,
        CONVERT(IFNULL(src.treatment_plan,'')        USING utf8mb4) COLLATE {C} AS treatment_plan,
        CONVERT(IFNULL(src.investigations,'')        USING utf8mb4) COLLATE {C} AS investigations,
        CONVERT(IFNULL(src.past_medical_history,'')  USING utf8mb4) COLLATE {C} AS past_medical_history,
        CONVERT(IFNULL(src.remarks,'')               USING utf8mb4) COLLATE {C} AS remarks,
        src.note_created_at,
        src.note_updated_at
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY visit, source_schema
                   ORDER BY note_updated_at DESC, id DESC
               ) AS rn
        FROM `{STAGING_SCHEMA}`.`evaluation_doctor_notes_merged`
        WHERE note_created_at >= '{DATA_CUTOFF_DATE}'
          {wm_filter}
    ) src
    WHERE src.rn = 1
    """


def build_stg_doctor_notes(engine: Engine, full: bool = False) -> None:
    table = "stg_doctor_notes"
    if full or not _table_exists(engine, table):
        _full_rebuild(engine, table, STG_DOCTOR_NOTES_DDL, _doctor_notes_select())
    else:
        for logical in SNOWFLAKE_SOURCES.keys():
            wm = _get_watermark(engine, table, "note_updated_at", logical)
            wm_filter = f"AND note_updated_at > '{wm}' AND source_schema = '{logical}'"
            _incremental(engine, table, _doctor_notes_select(wm_filter),
                         "note_updated_at", logical)


# ── stg_icd10 ─────────────────────────────────────────────────────────────────
# Also reads from STAGING.evaluation_doctor_notes_merged.

STG_ICD10_DDL = f"""(
    raw_note_id                 INT          NOT NULL,
    source_schema               VARCHAR(50)  NOT NULL,
    sk_visit_id                 VARCHAR(100),
    icd10_note_code             VARCHAR(20)  COLLATE {C},
    icd10_variation_code        VARCHAR(20)  COLLATE {C},
    icd10_variation_name        VARCHAR(500) COLLATE {C},
    icd10_type_code             VARCHAR(20)  COLLATE {C},
    icd10_type_name             VARCHAR(500) COLLATE {C},
    icd10_subcategory_name      VARCHAR(500) COLLATE {C},
    icd10_subcategory_code_from VARCHAR(20)  COLLATE {C},
    icd10_subcategory_code_to   VARCHAR(20)  COLLATE {C},
    icd10_category_name         VARCHAR(500) COLLATE {C},
    icd10_category_code_from    VARCHAR(20)  COLLATE {C},
    icd10_category_code_to      VARCHAR(20)  COLLATE {C},
    note_created_at             DATETIME,
    PRIMARY KEY (raw_note_id, source_schema),
    INDEX idx_visit_id (sk_visit_id),
    INDEX idx_source   (source_schema),
    INDEX idx_code     (icd10_note_code),
    INDEX idx_updated  (note_created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE={C}"""


def _icd10_select(wm_filter: str = "") -> str:
    return f"""
    SELECT
        CONCAT(src.source_schema, '|', src.visit)  AS sk_visit_id,
        src.source_schema,
        src.id                                      AS raw_note_id,
        CONVERT(IFNULL(src.note_code,'')                   USING utf8mb4) COLLATE {C} AS icd10_note_code,
        CONVERT(IFNULL(src.icd10_variation_code,'')        USING utf8mb4) COLLATE {C} AS icd10_variation_code,
        CONVERT(IFNULL(src.icd10_variation_name,'')        USING utf8mb4) COLLATE {C} AS icd10_variation_name,
        CONVERT(IFNULL(src.icd10_type_code,'')             USING utf8mb4) COLLATE {C} AS icd10_type_code,
        CONVERT(IFNULL(src.icd10_type_name,'')             USING utf8mb4) COLLATE {C} AS icd10_type_name,
        CONVERT(IFNULL(src.icd10_subcategory_name,'')      USING utf8mb4) COLLATE {C} AS icd10_subcategory_name,
        CONVERT(IFNULL(src.icd10_subcategory_code_from,'') USING utf8mb4) COLLATE {C} AS icd10_subcategory_code_from,
        CONVERT(IFNULL(src.icd10_subcategory_code_to,'')   USING utf8mb4) COLLATE {C} AS icd10_subcategory_code_to,
        CONVERT(IFNULL(src.icd10_category_name,'')         USING utf8mb4) COLLATE {C} AS icd10_category_name,
        CONVERT(IFNULL(src.icd10_category_code_from,'')    USING utf8mb4) COLLATE {C} AS icd10_category_code_from,
        CONVERT(IFNULL(src.icd10_category_code_to,'')      USING utf8mb4) COLLATE {C} AS icd10_category_code_to,
        src.note_created_at
    FROM `{STAGING_SCHEMA}`.`evaluation_doctor_notes_merged` src
    WHERE src.note_created_at >= '{DATA_CUTOFF_DATE}'
      AND src.note_code IS NOT NULL
      {wm_filter}
    """


def build_stg_icd10(engine: Engine, full: bool = False) -> None:
    table = "stg_icd10"
    if full or not _table_exists(engine, table):
        _full_rebuild(engine, table, STG_ICD10_DDL, _icd10_select())
    else:
        for logical in SNOWFLAKE_SOURCES.keys():
            wm = _get_watermark(engine, table, "note_created_at", logical)
            wm_filter = f"AND src.note_created_at > '{wm}' AND src.source_schema = '{logical}'"
            _incremental(engine, table, _icd10_select(wm_filter), "note_created_at", logical)


# ── stg_prescriptions ─────────────────────────────────────────────────────────

STG_RX_DDL = f"""(
    sk_prescription_id VARCHAR(100) NOT NULL,
    sk_visit_id        VARCHAR(100),
    source_schema      VARCHAR(50)  NOT NULL,
    raw_id             INT          NOT NULL,
    drug_name          VARCHAR(500) COLLATE {C},
    dosage             VARCHAR(200) COLLATE {C},
    quantity           INT,
    dispensed_quantity INT,
    stopped            INT,
    stop_reason        TEXT         COLLATE {C},
    price              DECIMAL(18,2),
    prescription_date  DATE,
    rx_created_at      DATETIME,
    rx_updated_at      DATETIME,
    PRIMARY KEY (sk_prescription_id),
    INDEX idx_visit   (sk_visit_id),
    INDEX idx_source  (source_schema),
    INDEX idx_updated (rx_updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE={C}"""


def _rx_select(src_schema: str, logical: str, wm_filter: str = "") -> str:
    return f"""
    SELECT
        CONCAT('{logical}', '|', src.id)     AS sk_prescription_id,
        CONCAT('{logical}', '|', src.visit)  AS sk_visit_id,
        '{logical}'                           AS source_schema,
        src.id                                AS raw_id,
        CONVERT(IFNULL(src.drug_name,'')   USING utf8mb4) COLLATE {C} AS drug_name,
        CONVERT(IFNULL(src.dosage,'')      USING utf8mb4) COLLATE {C} AS dosage,
        src.quantity,
        src.dispensed_quantity,
        src.stopped,
        CONVERT(IFNULL(src.stop_reason,'') USING utf8mb4) COLLATE {C} AS stop_reason,
        src.price,
        DATE(src.created_at)   AS prescription_date,
        src.created_at         AS rx_created_at,
        src.updated_at         AS rx_updated_at
    FROM `{src_schema}`.`evaluation_prescriptions` src
    WHERE src.created_at >= '{DATA_CUTOFF_DATE}'
      {wm_filter}
    """


def build_stg_prescriptions(engine: Engine, full: bool = False) -> None:
    table = "stg_prescriptions"
    if full or not _table_exists(engine, table):
        union_parts = [
            _rx_select(raw_schema(lg), lg)
            for lg in SNOWFLAKE_SOURCES.keys()
        ]
        _full_rebuild(engine, table, STG_RX_DDL,
                      "\nUNION ALL\n".join(union_parts))
    else:
        for logical in SNOWFLAKE_SOURCES.keys():
            wm = _get_watermark(engine, table, "rx_updated_at", logical)
            wm_filter = f"AND src.updated_at > '{wm}'"
            _incremental(engine, table,
                         _rx_select(raw_schema(logical), logical, wm_filter),
                         "rx_updated_at", logical)


# ── stg_invoices ──────────────────────────────────────────────────────────────

STG_INV_DDL = f"""(
    sk_invoice_id      VARCHAR(100) NOT NULL,
    sk_visit_id        VARCHAR(100),
    source_schema      VARCHAR(50)  NOT NULL,
    raw_id             INT          NOT NULL,
    invoice_no         VARCHAR(100) COLLATE {C},
    invoice_date       DATE,
    status             VARCHAR(50)  COLLATE {C},
    patient_id         INT,
    amount             DECIMAL(18,2),
    balance            DECIMAL(18,2),
    paid               INT,
    for_cash           INT,
    invoice_created_at DATETIME,
    invoice_updated_at DATETIME,
    PRIMARY KEY (sk_invoice_id),
    INDEX idx_visit   (sk_visit_id),
    INDEX idx_source  (source_schema),
    INDEX idx_updated (invoice_updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE={C}"""


def _invoices_select(src_schema: str, logical: str, wm_filter: str = "") -> str:
    return f"""
    SELECT
        CONCAT('{logical}', '|', src.id)     AS sk_invoice_id,
        CONCAT('{logical}', '|', src.visit)  AS sk_visit_id,
        '{logical}'                           AS source_schema,
        src.id                                AS raw_id,
        CONVERT(IFNULL(src.invoice_no,'') USING utf8mb4) COLLATE {C} AS invoice_no,
        src.invoice_date,
        CONVERT(IFNULL(src.status,'')    USING utf8mb4) COLLATE {C} AS status,
        src.patient_id,
        src.amount,
        src.balance,
        src.paid,
        src.for_cash,
        src.created_at  AS invoice_created_at,
        src.updated_at  AS invoice_updated_at
    FROM `{src_schema}`.`finance_invoices` src
    WHERE src.deleted_at IS NULL
      {wm_filter}
    """


def build_stg_invoices(engine: Engine, full: bool = False) -> None:
    table = "stg_invoices"
    if full or not _table_exists(engine, table):
        union_parts = [
            _invoices_select(raw_schema(lg), lg)
            for lg in SNOWFLAKE_SOURCES.keys()
        ]
        _full_rebuild(engine, table, STG_INV_DDL,
                      "\nUNION ALL\n".join(union_parts))
    else:
        for logical in SNOWFLAKE_SOURCES.keys():
            wm = _get_watermark(engine, table, "invoice_updated_at", logical)
            wm_filter = f"AND src.updated_at > '{wm}'"
            _incremental(engine, table,
                         _invoices_select(raw_schema(logical), logical, wm_filter),
                         "invoice_updated_at", logical)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_staging(engine: Engine, full: bool = False) -> None:
    log.info("=" * 60)
    log.info("Step 2: Staging (%s)", "FULL" if full else "INCREMENTAL")
    log.info("=" * 60)

    _ensure_staging_schema(engine)

    build_stg_patients(engine, full)
    build_stg_visits(engine, full)
    build_stg_doctor_notes(engine, full)
    build_stg_icd10(engine, full)
    build_stg_prescriptions(engine, full)
    build_stg_invoices(engine, full)

    log.info("=" * 60)
    log.info("Step 2 complete")
    log.info("=" * 60)