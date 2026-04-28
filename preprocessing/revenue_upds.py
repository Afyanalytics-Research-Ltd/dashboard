"""Stack every <db>.*_CLEAN.<source_table> into <db>.STAGING.<staging_name>,
then aggregate by patient into <db>.REPORTING.

Pipeline (matches the 4-layer architecture):

    Layer 0  Raw            <db>.*_CLEAN.<source_table>  (and TENRI)
    Layer 1  Stage          <db>.STAGING.STG_FIN_* (one row per source row)
    Layer 2  Atomic fact    <db>.REPORTING.FACT_PATIENT_FINANCE_EVENTS
    Layer 3  Visit fact     <db>.REPORTING.FACT_PATIENT_INVOICE

The script is defensive: any individual schema, table, or layer that fails
is logged with full context and skipped — the run keeps going. A summary at
the end shows what succeeded and what didn't.

Setup:
    pip install snowflake-connector-python

    export SNOWFLAKE_USER=...
    export SNOWFLAKE_PASSWORD=...        # or SNOWFLAKE_PRIVATE_KEY_PATH
    export SNOWFLAKE_ACCOUNT=xy12345.eu-west-1
    export SNOWFLAKE_WAREHOUSE=COMPUTE_WH
    export SNOWFLAKE_DATABASE=YOUR_DB
    export SNOWFLAKE_ROLE=DATAANALYSTS
    # optional:
    export TARGET_SCHEMA=STAGING            # default
    export REPORTING_SCHEMA=REPORTING       # default
    export EXTRA_SCHEMAS=TENRI,SOMETHING    # comma-separated, alongside *_CLEAN
    export SF_DRY_RUN=1                     # print SQL but don't execute
    export SF_LOG_LEVEL=DEBUG               # default INFO

Run:
    SF_DRY_RUN=1 python stack_clean_schemas.py    # preview
    python stack_clean_schemas.py                 # build for real
"""
from __future__ import annotations

import logging
import os
import sys
import traceback

import snowflake.connector
from snowflake.connector.errors import ProgrammingError, DatabaseError

logging.basicConfig(
    level=os.environ.get("SF_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(message)s",
)
log = logging.getLogger("stack_clean")


# Schemas that should always be included even though they don't match
# the *_CLEAN naming convention.
ALWAYS_INCLUDE_SCHEMAS = {"TENRI"}


# --------------------------------------------------------------------- spec
def col(alias, expr=None, requires=None):
    expr = expr or alias
    if requires is None:
        requires = [expr.split("::")[0].strip()]
    return (alias, expr, requires)


SOURCES = {
    "STG_FIN_INVOICES": {
        "source_table": "FINANCE_INVOICES",
        "columns": [
            col("PATIENT_ID"),
            col("INVOICE_NO"),
            col("INVOICE_NO_PREFIX"),
            col("INVOICE_DATE"),
            col("ACTUAL_INVOICE_CREATION_DATE"),
            col("AMOUNT"),
            col("EXEMPTION_AMOUNT"),
            col("PAID"),
            col("CLAIM_NO"),
            col("SPLIT_BILL_ID"),
            col("UNBILLED_BY"),
        ],
    },
    "STG_FIN_EVALUATION_PAYMENTS": {
        "source_table": "FINANCE_EVALUATION_PAYMENTS",
        "columns": [
            col("PATIENT_ID", "PATIENT::VARCHAR", ["PATIENT"]),
            col("INVOICE_ID"),
            col("AMOUNT"),
            col("CASH_AMOUNT"),
            col("CARD_AMOUNT"),
            col("CHEQUE_AMOUNT"),
            col("MPESA_AMOUNT"),
            col("JAMBOPAY_AMOUNT"),
            col("PATIENTACCOUNT_AMOUNT"),
            col("TOTAL_AMOUNT_WAS"),
            col("TOTAL_CASH_PAYMENT"),
        ],
    },
    "STG_FIN_TRANSACTIONS": {
        "source_table": "FINANCE_TRANSACTIONS",
        "columns": [
            col("PATIENT_ID"),
            col("INVOICE_ID"),
            col("BILL_REFERENCE"),
            col("AMOUNT"),
            col("EXT_PAID_AMOUNT"),
        ],
    },
    "STG_FIN_PATIENT_DEPOSITS": {
        "source_table": "FINANCE_PATIENT_DEPOSITS",
        "columns": [
            col("PATIENT_ID"),
            col("AMOUNT"),
            col("PAYMENT_MODE"),
        ],
    },
    "STG_FIN_PATIENT_INVOICES": {
        "source_table": "FINANCE_PATIENT_INVOICES",
        "columns": [
            col("PATIENT_ID"),
            col("INVOICE_NO"),
            col("AMOUNT"),
        ],
    },
    "STG_FIN_WAIVERS": {
        "source_table": "FINANCE_WAIVERS",
        "columns": [
            col("PATIENT_ID"),
            col("INVOICE_ID"),
            col("AMOUNT"),
        ],
    },
    "STG_FIN_PATIENT_ACCOUNTS": {
        "source_table": "FINANCE_PATIENT_ACCOUNTS",
        "columns": [
            col("PATIENT_ID", "PATIENT::VARCHAR", ["PATIENT"]),
            col("PATIENT_DEPOSIT_ID"),
            col("PATIENT_WITHDRAWAL_ID"),
            col("AMOUNT_BEFORE"),
            col("AMOUNT_AFTER"),
        ],
    },
    "STG_INVENTORY_PRODUCT_SALES": {
        "source_table": "INVENTORY_INVENTORY_BATCH_PRODUCT_SALES",
        "columns": [
            col("PATIENT_ID", "PATIENT::VARCHAR", ["PATIENT"]),
            col("AMOUNT"),
            col("PAYMENT_MODE"),
        ],
    },
    "STG_RECEPTION_PATIENT_INSURANCE": {
        "source_table": "RECEPTION_PATIENT_INSURANCE",
        "columns": [
            col("PATIENT_ID", "PATIENT::VARCHAR", ["PATIENT"]),
            col("SCHEMES_AMOUNT"),
            col("SCHEMES_CAPITATION_COPAY_AMOUNT"),
        ],
    },
    "STG_INPATIENT_ADMISSIONS": {
        "source_table": "INPATIENT_ADMISSIONS",
        "columns": [
            col("PATIENT_ID"),
            col("IS_DISCHARGED"),
            col("DISCHARGED_AT"),
            col("PAYMENT_MODE"),
        ],
    },
}


# ---------------------------------------------------------------- snowflake
def get_connection():
    log.info("Opening Snowflake connection ...")
    log.debug("  user=%s account=%s warehouse=%s database=%s role=%s",
              os.getenv("SNOWFLAKE_USER"),
              os.getenv("SNOWFLAKE_ACCOUNT"),
              os.getenv("SNOWFLAKE_WAREHOUSE"),
              os.getenv("SNOWFLAKE_DATABASE"),
              os.getenv("SNOWFLAKE_ROLE"))
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER").strip(),
        account=os.getenv("SNOWFLAKE_ACCOUNT").strip(),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE").strip(),
        database=os.getenv("SNOWFLAKE_DATABASE").strip(),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC").strip(),
        role=os.getenv("SNOWFLAKE_ROLE", "public").strip(),
        private_key_file=os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH").strip(),
    )


def log_session_state(cur):
    """Print what Snowflake actually sees so connection mistakes are obvious."""
    try:
        cur.execute("""
            SELECT CURRENT_USER(), CURRENT_ROLE(),
                   CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA()
        """)
        user, role, wh, db, sch = cur.fetchone()
        log.info("Session: user=%s role=%s warehouse=%s database=%s schema=%s",
                 user, role, wh, db, sch)
    except Exception as e:
        log.warning("Could not read session state: %s", e)


def list_clean_schemas(cur, database):
    """Return *_CLEAN schemas plus any schema in ALWAYS_INCLUDE_SCHEMAS or
    EXTRA_SCHEMAS env var, but only if they actually exist in the database.
    """
    log.info("Discovering source schemas in %s ...", database)
    try:
        cur.execute(
            f"""
            SELECT schema_name
            FROM   {database}.INFORMATION_SCHEMA.SCHEMATA
            ORDER  BY schema_name
            """
        )
        all_schemas = [r[0] for r in cur.fetchall()]
    except Exception:
        log.exception("Failed to list schemas in %s; aborting.", database)
        return []

    log.debug("All schemas in %s: %s", database, all_schemas)
    existing_upper = {s.upper(): s for s in all_schemas}

    extra_env = os.environ.get("EXTRA_SCHEMAS", "")
    extras = {s.strip().upper() for s in extra_env.split(",") if s.strip()}
    must_include = (ALWAYS_INCLUDE_SCHEMAS | extras)

    selected = []
    for schema in all_schemas:
        if schema.upper().endswith("_CLEAN"):
            selected.append(schema)
            log.debug("  + auto-discovered _CLEAN schema: %s", schema)

    for name in sorted(must_include):
        if name in existing_upper and existing_upper[name] not in selected:
            selected.append(existing_upper[name])
            log.info("  + including non-_CLEAN schema: %s", existing_upper[name])
        elif name not in existing_upper:
            log.warning("  ! requested schema %s not found in %s; skipping",
                        name, database)

    return selected


def fetch_columns(cur, database, schema, table):
    """Return the set of columns present in <db>.<schema>.<table>, or empty
    set if the table doesn't exist or we can't read it.
    """
    try:
        cur.execute(
            f"""
            SELECT column_name
            FROM   {database}.INFORMATION_SCHEMA.COLUMNS
            WHERE  table_schema = %s AND table_name = %s
            """,
            (schema, table),
        )
        cols = {r[0].upper() for r in cur.fetchall()}
        log.debug("    columns in %s.%s.%s: %d found", database, schema, table, len(cols))
        return cols
    except Exception as e:
        log.warning("    could not read columns for %s.%s.%s: %s",
                    database, schema, table, e)
        return set()


# ============================================================ Layer 1: STAGING
def build_select(database, schema, source_table, columns, present):
    parts = []
    for alias, expr, requires in columns:
        if all(r.upper() in present for r in requires):
            parts.append(f"    {expr} AS {alias}")
        else:
            parts.append(f"    NULL AS {alias}")
    parts.append(f"    '{source_table}' AS SOURCE_TABLE")
    parts.append(f"    '{schema}' AS SOURCE_SCHEMA")
    return (
        "SELECT\n" + ",\n".join(parts)
        + f"\nFROM {database}.{schema}.{source_table}"
    )


def build_stg(staging_name, database, target_schema, schemas, spec, get_cols):
    selects = []
    skipped = []
    for schema in schemas:
        present = get_cols(schema)
        if not present:
            log.info("  - %s.%s missing or unreadable - skipping",
                     schema, spec["source_table"])
            skipped.append(schema)
            continue
        selects.append(
            build_select(database, schema, spec["source_table"],
                         spec["columns"], present)
        )
        log.debug("  + included %s.%s in %s",
                  schema, spec["source_table"], staging_name)
    if not selects:
        return None, skipped
    body = "\n\nUNION ALL\n\n".join(selects)
    ddl = f"CREATE OR REPLACE TABLE {database}.{target_schema}.{staging_name} AS\n{body}"
    return ddl, skipped


# ====================================================== Layer 2: atomic events
def build_fact_patient_finance_events(database, staging, target):
    return f"""CREATE OR REPLACE TABLE {database}.{target}.FACT_PATIENT_FINANCE_EVENTS AS
SELECT
    PATIENT_ID::VARCHAR              AS PATIENT_ID,
    'INVOICE_ISSUED'                 AS EVENT_TYPE,
    AMOUNT                           AS AMOUNT,
    NULL                             AS INVOICE_ID,
    INVOICE_NO                       AS INVOICE_NO,
    NULL                             AS PAYMENT_MODE,
    INVOICE_DATE                     AS EVENT_DATE,
    SOURCE_TABLE,
    SOURCE_SCHEMA
FROM {database}.{staging}.STG_FIN_INVOICES

UNION ALL

SELECT PATIENT_ID::VARCHAR, 'EVALUATION_PAYMENT', AMOUNT, INVOICE_ID, NULL, NULL, NULL, SOURCE_TABLE, SOURCE_SCHEMA
FROM {database}.{staging}.STG_FIN_EVALUATION_PAYMENTS

UNION ALL

SELECT PATIENT_ID::VARCHAR, 'TRANSACTION', AMOUNT, INVOICE_ID, NULL, NULL, NULL, SOURCE_TABLE, SOURCE_SCHEMA
FROM {database}.{staging}.STG_FIN_TRANSACTIONS

UNION ALL

SELECT PATIENT_ID::VARCHAR, 'DEPOSIT', AMOUNT, NULL, NULL, PAYMENT_MODE, NULL, SOURCE_TABLE, SOURCE_SCHEMA
FROM {database}.{staging}.STG_FIN_PATIENT_DEPOSITS

UNION ALL

SELECT PATIENT_ID::VARCHAR, 'PATIENT_INVOICE', AMOUNT, NULL, INVOICE_NO, NULL, NULL, SOURCE_TABLE, SOURCE_SCHEMA
FROM {database}.{staging}.STG_FIN_PATIENT_INVOICES

UNION ALL

SELECT PATIENT_ID::VARCHAR, 'WAIVER', AMOUNT, INVOICE_ID, NULL, NULL, NULL, SOURCE_TABLE, SOURCE_SCHEMA
FROM {database}.{staging}.STG_FIN_WAIVERS

UNION ALL

SELECT PATIENT_ID::VARCHAR, 'ACCOUNT_DELTA',
       COALESCE(AMOUNT_AFTER, 0) - COALESCE(AMOUNT_BEFORE, 0),
       NULL, NULL, NULL, NULL, SOURCE_TABLE, SOURCE_SCHEMA
FROM {database}.{staging}.STG_FIN_PATIENT_ACCOUNTS

UNION ALL

SELECT PATIENT_ID::VARCHAR, 'PRODUCT_SALE', AMOUNT, NULL, NULL, PAYMENT_MODE, NULL, SOURCE_TABLE, SOURCE_SCHEMA
FROM {database}.{staging}.STG_INVENTORY_PRODUCT_SALES

UNION ALL

SELECT PATIENT_ID::VARCHAR, 'INSURANCE_SCHEME', SCHEMES_AMOUNT, NULL, NULL, NULL, NULL, SOURCE_TABLE, SOURCE_SCHEMA
FROM {database}.{staging}.STG_RECEPTION_PATIENT_INSURANCE

UNION ALL

SELECT PATIENT_ID::VARCHAR, 'ADMISSION', NULL, NULL, NULL, PAYMENT_MODE, DISCHARGED_AT, SOURCE_TABLE, SOURCE_SCHEMA
FROM {database}.{staging}.STG_INPATIENT_ADMISSIONS"""


# ===================================================== Layer 3: per-invoice fact
def build_fact_patient_invoice(database, target):
    return f"""CREATE OR REPLACE TABLE {database}.{target}.FACT_PATIENT_INVOICE AS
WITH events AS (
    SELECT * FROM {database}.{target}.FACT_PATIENT_FINANCE_EVENTS
    WHERE INVOICE_ID IS NOT NULL OR INVOICE_NO IS NOT NULL
)
SELECT
    PATIENT_ID,
    COALESCE(INVOICE_ID::VARCHAR, INVOICE_NO::VARCHAR)              AS INVOICE_KEY,
    ANY_VALUE(SOURCE_SCHEMA)                                        AS FACILITY_KEY,
    SUM(CASE WHEN EVENT_TYPE = 'INVOICE_ISSUED'    THEN AMOUNT ELSE 0 END) AS TOTAL_BILLED,
    SUM(CASE WHEN EVENT_TYPE = 'PATIENT_INVOICE'   THEN AMOUNT ELSE 0 END) AS TOTAL_PATIENT_INVOICE,
    SUM(CASE WHEN EVENT_TYPE = 'EVALUATION_PAYMENT' THEN AMOUNT ELSE 0 END) AS TOTAL_PAID_EVAL,
    SUM(CASE WHEN EVENT_TYPE = 'TRANSACTION'       THEN AMOUNT ELSE 0 END) AS TOTAL_TRANSACTIONS,
    SUM(CASE WHEN EVENT_TYPE = 'WAIVER'            THEN AMOUNT ELSE 0 END) AS TOTAL_WAIVED,
    (SUM(CASE WHEN EVENT_TYPE = 'INVOICE_ISSUED' THEN AMOUNT ELSE 0 END)
     - SUM(CASE WHEN EVENT_TYPE IN ('EVALUATION_PAYMENT', 'TRANSACTION', 'WAIVER') THEN AMOUNT ELSE 0 END)
    )                                                               AS BALANCE_DUE,
    COUNT(*)                                                        AS EVENT_COUNT,
    MIN(EVENT_DATE)                                                 AS FIRST_EVENT_DATE,
    MAX(EVENT_DATE)                                                 AS LAST_EVENT_DATE
FROM events
WHERE PATIENT_ID IS NOT NULL
GROUP BY PATIENT_ID, COALESCE(INVOICE_ID::VARCHAR, INVOICE_NO::VARCHAR)"""


# ============================================================ orchestration
def safe_execute(cur, sql, label):
    """Run a SQL statement with logging. Returns True on success, False on failure.

    Never raises — caller decides whether to keep going. The full SQL is dumped
    at DEBUG so re-running with SF_LOG_LEVEL=DEBUG shows exactly what failed.
    """
    log.debug("--- SQL [%s] ---\n%s", label, sql)
    try:
        cur.execute(sql)
        return True
    except (ProgrammingError, DatabaseError) as e:
        log.error("SQL failed for %s: %s", label, e)
        log.debug("Failing SQL was:\n%s", sql)
        return False
    except Exception:
        log.exception("Unexpected failure for %s", label)
        log.debug("Failing SQL was:\n%s", sql)
        return False


def run_layer(cur, name, sql, database, target, dry_run):
    """Build one table. Returns True/False so the caller can track results."""
    log.info("Building %s.%s.%s ...", database, target, name)
    if dry_run:
        print(f"\n-- {name}\n{sql};\n")
        return True

    if not safe_execute(cur, sql, label=f"{target}.{name}"):
        log.warning("  -> skipping %s due to error above", name)
        return False

    try:
        cur.execute(f"SELECT COUNT(*) FROM {database}.{target}.{name}")
        n = cur.fetchone()[0]
        log.info("  -> %s.%s.%s built: %s rows", database, target, name, n)
    except Exception as e:
        log.warning("  -> built %s but could not read row count: %s", name, e)
    return True


def main():
    database = os.environ.get("SNOWFLAKE_DATABASE")
    staging = os.environ.get("TARGET_SCHEMA", "STAGING")
    reporting = os.environ.get("REPORTING_SCHEMA", "REPORTING")
    dry_run = os.environ.get("SF_DRY_RUN", "0") in ("1", "true", "yes")

    if not database:
        log.error("SNOWFLAKE_DATABASE is required.")
        return 2

    log.info("=" * 60)
    log.info("Pipeline starting (dry_run=%s)", dry_run)
    log.info("  database=%s staging=%s reporting=%s", database, staging, reporting)
    log.info("=" * 60)

    try:
        conn = get_connection()
    except Exception:
        log.exception("Could not open Snowflake connection; aborting.")
        return 1

    summary = {"layer1_built": [], "layer1_failed": [], "layer1_skipped": [],
               "layer2": None, "layer3": None}

    try:
        cur = conn.cursor()
        log_session_state(cur)

        schemas = list_clean_schemas(cur, database)
        if not schemas:
            log.error("No source schemas found; aborting.")
            return 1
        log.info("Found %d schema(s): %s", len(schemas), ", ".join(schemas))

        if not dry_run:
            for sch in (staging, reporting):
                ok = safe_execute(
                    cur,
                    f"CREATE SCHEMA IF NOT EXISTS {database}.{sch}",
                    label=f"create_schema_{sch}",
                )
                if not ok:
                    log.warning("Could not ensure schema %s.%s exists; "
                                "subsequent steps may fail.", database, sch)

        # ---------- Layer 1: STAGING ----------
        log.info("-" * 60)
        log.info("Layer 1: building staging tables")
        log.info("-" * 60)
        for staging_name, spec in SOURCES.items():
            log.info("Preparing %s (source: %s)", staging_name, spec["source_table"])
            try:
                ddl, skipped_schemas = build_stg(
                    staging_name, database, staging, schemas, spec,
                    lambda s, t=spec["source_table"]: fetch_columns(cur, database, s, t),
                )
            except Exception:
                log.exception("Could not assemble DDL for %s; skipping.", staging_name)
                summary["layer1_failed"].append(staging_name)
                continue

            if ddl is None:
                log.warning("  no source data found anywhere for %s; skipping.",
                            staging_name)
                summary["layer1_skipped"].append(staging_name)
                continue

            if skipped_schemas:
                log.info("  (skipped schemas for this table: %s)",
                         ", ".join(skipped_schemas))

            ok = run_layer(cur, staging_name, ddl, database, staging, dry_run)
            if ok:
                summary["layer1_built"].append(staging_name)
            else:
                summary["layer1_failed"].append(staging_name)

        # ---------- Layer 2: atomic events ----------
        log.info("-" * 60)
        log.info("Layer 2: FACT_PATIENT_FINANCE_EVENTS")
        log.info("-" * 60)
        try:
            ok2 = run_layer(
                cur, "FACT_PATIENT_FINANCE_EVENTS",
                build_fact_patient_finance_events(database, staging, reporting),
                database, reporting, dry_run,
            )
            summary["layer2"] = ok2
        except Exception:
            log.exception("Layer 2 raised; skipping.")
            summary["layer2"] = False

        # ---------- Layer 3: per-invoice fact ----------
        log.info("-" * 60)
        log.info("Layer 3: FACT_PATIENT_INVOICE")
        log.info("-" * 60)
        if summary["layer2"]:
            try:
                ok3 = run_layer(
                    cur, "FACT_PATIENT_INVOICE",
                    build_fact_patient_invoice(database, reporting),
                    database, reporting, dry_run,
                )
                summary["layer3"] = ok3
            except Exception:
                log.exception("Layer 3 raised; skipping.")
                summary["layer3"] = False
        else:
            log.warning("Layer 2 did not build successfully; skipping Layer 3.")
            summary["layer3"] = False

    finally:
        try:
            conn.close()
            log.debug("Connection closed.")
        except Exception:
            log.warning("Failed to close Snowflake connection cleanly.")

    # ---------- Summary ----------
    log.info("=" * 60)
    log.info("Pipeline summary")
    log.info("=" * 60)
    log.info("Layer 1 built   (%d): %s",
             len(summary["layer1_built"]), summary["layer1_built"])
    log.info("Layer 1 failed  (%d): %s",
             len(summary["layer1_failed"]), summary["layer1_failed"])
    log.info("Layer 1 skipped (%d): %s",
             len(summary["layer1_skipped"]), summary["layer1_skipped"])
    log.info("Layer 2 (FACT_PATIENT_FINANCE_EVENTS): %s",
             "OK" if summary["layer2"] else "FAILED/SKIPPED")
    log.info("Layer 3 (FACT_PATIENT_INVOICE):        %s",
             "OK" if summary["layer3"] else "FAILED/SKIPPED")
    log.info("Done.")

    failures = len(summary["layer1_failed"]) \
        + (0 if summary["layer2"] else 1) \
        + (0 if summary["layer3"] else 1)
    return 0 if failures == 0 else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        log.error("Fatal error in main():\n%s", traceback.format_exc())
        sys.exit(1)