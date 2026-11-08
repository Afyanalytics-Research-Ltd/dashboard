# dags/docs_to_events_raw.py
"""
docs_to_events_raw — bridge {facility}_DOCS.EXTRACTED_FORMS → {facility}_RAW.EVENTS_RAW

Writes V3-shaped EVENTS_RAW rows per extracted record:
  source_table='patients',          module_source='reception-service'
  source_table='visits',            module_source='reception-service'
  source_table='vitals',            module_source='patient-evaluation-service'
  source_table='doctor_notes',      module_source='patient-evaluation-service'
  source_table='inv_products',      module_source='inventory-service'  (when default_store in conf)
  source_table='inv_stock_movements', module_source='inventory-service'  (when default_store in conf)

Requires 'facility' in DAG conf (e.g. {"facility": "siaya"}).
Derives source/destination schemas from that key — no code changes per new facility.

snowflake_to_v3_migration.discover_tables() reads module_source from EVENTS_RAW
and routes reception-service rows through v3_passthrough automatically.
v3_passthrough handles: name splitting (last_name → first+last), patient_no
generation (MIG-{FACILITY}-{id}), visit_no generation, FK resolution, clinic fill.

Zero changes to v3_passthrough, snowflake_to_v3_migration, or v2_to_v3_api_migration.

Run order: gdrive_landingai_extract → THIS DAG → snowflake_flatten_clean → snowflake_to_v3_migration
"""
from __future__ import annotations

import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.exceptions import AirflowFailException
from airflow.sdk import Param, dag, get_current_context, task

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipelines_common import lap_callback, use_pipelines_dir

log = logging.getLogger(__name__)

DAG_ID = "docs_to_events_raw"
MODULE_SOURCE = "reception-service"


def _require_facility(facility: str) -> None:
    if not facility or not facility.replace('-', '').replace('_', '').isalnum():
        raise AirflowFailException(
            "docs_to_events_raw requires 'facility' in DAG conf "
            "(e.g. {\"facility\": \"siaya\"}). Got: " + repr(facility)
        )


def run_setup(cur, facility: str, dst: str) -> None:
    """Create source and destination schemas and EVENTS_RAW table if they do not exist."""
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS HOSPITALS.{facility.upper()}_DOCS")
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS HOSPITALS.{facility.upper()}_RAW")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {dst} (
            facility_id    VARCHAR,
            ingested_at    TIMESTAMP_TZ,
            module_source  VARCHAR,
            source_table   VARCHAR,
            namespace      VARCHAR,
            payload        VARIANT
        )
    """)


def check_null_names(cur, src: str) -> int:
    """Return count of rows with NULL patient_name and log a warning if any.

    NULL names → v3_passthrough holds the patient back (first_name required);
    the visit is also held back until its patient_id resolves.
    """
    try:
        row = cur.execute(f"SELECT COUNT(*) FROM {src} WHERE EXTRACTION:patient_name::STRING IS NULL").fetchone()
    except Exception as exc:
        # EXTRACTION column absent means step 1 wrote no VARIANT data — warn, don't block validate
        log.warning("check_null_names could not query %s (EXTRACTION column may be absent): %s", src, exc)
        return 0
    count = int(row[0]) if row else 0
    if count:
        log.warning(
            "%d row(s) in %s have NULL patient_name. Those patient records will be "
            "held back by v3_passthrough (first_name NOT NULL). Their visit rows will "
            "also be held back until the patient lands in V3.",
            count, src,
        )
    return count


def run_validate(cur, src: str, facility: str) -> int:
    """Return total row count; raise AirflowFailException if EXTRACTED_FORMS is empty."""
    try:
        row = cur.execute(f"SELECT COUNT(*) FROM {src}").fetchone()
    except Exception as e:
        msg = str(e)
        if "does not exist" in msg or "not authorized" in msg:
            raise AirflowFailException(
                f"{src} does not exist or is not accessible. "
                f"Step 1 (gdrive_landingai_extract) must run successfully before this step. "
                f"Check: (1) step 1 completed without error, (2) its conf included "
                f"push_to_snowflake=true, snowflake_schema={facility.upper()}_DOCS, snowflake_table=EXTRACTED_FORMS."
            ) from e
        raise
    count = int(row[0]) if row else 0
    if not count:
        raise AirflowFailException(
            f"{src} is empty. gdrive_landingai_extract must run first and write records here. "
            f"If it ran but this table is still empty, check: (1) the GDRIVE_SOURCE_FOLDER_ID "
            f"Airflow Variable points at the correct {facility} PDF folder, and (2) the DAG conf "
            f"passed snowflake_schema={facility.upper()}_DOCS and snowflake_table=EXTRACTED_FORMS."
        )
    check_null_names(cur, src)
    log.info("%s: %d row(s) to process", src, count)
    return count


def run_write(cur, src: str, dst: str, facility: str, default_store: str = '') -> dict:
    """DELETE stale rows then INSERT reception + evaluation (+ inventory if default_store) rows atomically."""
    if default_store and not re.fullmatch(
        r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
        default_store, re.IGNORECASE,
    ):
        raise AirflowFailException(f"default_store must be a UUID, got: {default_store!r}")
    delete_sql = f"""
        DELETE FROM {dst}
        WHERE facility_id = '{facility}'
          AND source_table IN ('patients', 'visits', 'vitals', 'doctor_notes', 'inv_products', 'inv_stock_movements')
          AND payload:file_id::STRING IN (SELECT FILE_ID FROM {src})
    """
    # Full name goes in last_name; v3_passthrough._patient_names() splits it → first + last.
    patient_insert_sql = f"""
        INSERT INTO {dst} (facility_id, ingested_at, module_source, source_table, namespace, payload)
        SELECT
            '{facility}',
            CURRENT_TIMESTAMP(),
            '{MODULE_SOURCE}',
            'patients',
            NULL,
            OBJECT_CONSTRUCT(
                'id',            RECORD_KEY,
                'file_id',       FILE_ID,
                'last_name',     EXTRACTION:patient_name::STRING,
                'patient_no',    COALESCE(EXTRACTION:camp_file_number::STRING, EXTRACTION:patient_id_file_no::STRING),
                'date_of_birth', TRY_TO_DATE(NULLIF(EXTRACTION:date_of_birth::STRING, ''), 'DD/MM/YYYY'),
                'age',           TRY_CAST(SPLIT_PART(EXTRACTION:age::STRING, ' ', 1) AS INTEGER),
                'gender',        EXTRACTION:sex::STRING,
                'phone',         EXTRACTION:phone_number::STRING,
                'address',       EXTRACTION:address::STRING
            )
        FROM {src}
    """
    # patient_id = RECORD_KEY links to the patient row above.
    # v3_passthrough holds this row back until the patient is confirmed in V3 (HOLD_IF_MISSING).
    visit_insert_sql = f"""
        INSERT INTO {dst} (facility_id, ingested_at, module_source, source_table, namespace, payload)
        SELECT
            '{facility}',
            CURRENT_TIMESTAMP(),
            '{MODULE_SOURCE}',
            'visits',
            NULL,
            OBJECT_CONSTRUCT(
                'id',         RECORD_KEY || '_v',
                'file_id',    FILE_ID,
                'patient_id', RECORD_KEY,
                'date',       TRY_TO_DATE(NULLIF(EXTRACTION:visit_date::STRING, ''), 'DD/MM/YYYY'),
                'notes',      COALESCE(EXTRACTION:nurses_triage_notes::STRING, EXTRACTION:history_of_present_illness::STRING, EXTRACTION:doctors_notes::STRING)
            )
        FROM {src}
    """
    vitals_insert_sql = f"""
        INSERT INTO {dst} (facility_id, ingested_at, module_source, source_table, namespace, payload)
        SELECT
            '{facility}',
            CURRENT_TIMESTAMP(),
            'patient-evaluation-service',
            'vitals',
            NULL,
            OBJECT_CONSTRUCT(
                'id',                RECORD_KEY || '_vitals',
                'file_id',           FILE_ID,
                'visit_id',          RECORD_KEY || '_v',
                'patient_id',        RECORD_KEY,
                'temperature',       TRY_CAST(SPLIT_PART(EXTRACTION:temperature::STRING, ' ', 1) AS FLOAT),
                'systolic_bp',       TRY_CAST(SPLIT_PART(EXTRACTION:blood_pressure::STRING, '/', 1) AS INTEGER),
                'diastolic_bp',      TRY_CAST(SPLIT_PART(SPLIT_PART(EXTRACTION:blood_pressure::STRING, '/', 2), ' ', 1) AS INTEGER),
                'pulse',             TRY_CAST(SPLIT_PART(EXTRACTION:heart_rate::STRING, ' ', 1) AS INTEGER),
                'weight',            TRY_CAST(SPLIT_PART(EXTRACTION:weight::STRING, ' ', 1) AS FLOAT),
                'height',            TRY_CAST(EXTRACTION:height::STRING AS FLOAT),
                'bmi',               TRY_CAST(EXTRACTION:body_mass_index::STRING AS FLOAT),
                'respiratory_rate',  TRY_CAST(SPLIT_PART(EXTRACTION:respiratory_rate::STRING, ' ', 1) AS INTEGER),
                'oxygen_saturation', TRY_CAST(SPLIT_PART(EXTRACTION:oxygen_saturation::STRING, ' ', 1) AS FLOAT)
            )
        FROM {src}
    """
    doctor_notes_insert_sql = f"""
        INSERT INTO {dst} (facility_id, ingested_at, module_source, source_table, namespace, payload)
        SELECT
            '{facility}',
            CURRENT_TIMESTAMP(),
            'patient-evaluation-service',
            'doctor_notes',
            NULL,
            OBJECT_CONSTRUCT(
                'id',                        RECORD_KEY || '_dn',
                'file_id',                   FILE_ID,
                'visit_id',                  RECORD_KEY || '_v',
                'patient_id',                RECORD_KEY,
                'chief_complaints',          ARRAY_TO_STRING(EXTRACTION:chief_complaints::ARRAY, ', '),
                'history_of_present_illness', EXTRACTION:history_of_present_illness::STRING,
                'examination',               NULLIF(TRIM(CONCAT_WS(' | ',
                                                 NULLIF(EXTRACTION:general_appearance::STRING, ''),
                                                 NULLIF(EXTRACTION:systemic_exam_cvs::STRING, ''),
                                                 NULLIF(EXTRACTION:systemic_exam_resp::STRING, ''),
                                                 NULLIF(EXTRACTION:systemic_exam_git::STRING, ''),
                                                 NULLIF(EXTRACTION:systemic_exam_cns::STRING, ''),
                                                 NULLIF(EXTRACTION:systemic_exam_msk_skin::STRING, '')
                                             )), ''),
                'surgical_history',          EXTRACTION:past_medical_surgical_history::STRING,
                'plan',                      EXTRACTION:doctors_notes::STRING
            )
        FROM {src}
    """
    inv_products_insert_sql = f"""
        INSERT INTO {dst} (facility_id, ingested_at, module_source, source_table, namespace, payload)
        SELECT
            '{facility}',
            CURRENT_TIMESTAMP(),
            'inventory-service',
            'inv_products',
            NULL,
            OBJECT_CONSTRUCT(
                'id',           RECORD_KEY || '_p_' || f.value:sn::STRING,
                'file_id',      FILE_ID,
                'generic_name', f.value:item_name::STRING
            )
        FROM {src},
        LATERAL FLATTEN(input => EXTRACTION:stock_items) f
        WHERE EXTRACTION:stock_items IS NOT NULL
    """
    inv_stock_movements_insert_sql = f"""
        INSERT INTO {dst} (facility_id, ingested_at, module_source, source_table, namespace, payload)
        SELECT
            '{facility}',
            CURRENT_TIMESTAMP(),
            'inventory-service',
            'inv_stock_movements',
            NULL,
            OBJECT_CONSTRUCT(
                'id',            RECORD_KEY || '_sm_' || f.value:sn::STRING,
                'file_id',       FILE_ID,
                'product_id',    RECORD_KEY || '_p_' || f.value:sn::STRING,
                'from_store_id', '{default_store}',
                'quantity',      TRY_CAST(SPLIT_PART(f.value:quantity::STRING, ' ', 1) AS FLOAT),
                'batch_no',      f.value:batch_no::STRING,
                'expiry_date',   f.value:expiry_date::STRING,
                'reason',        'opening_stock'
            )
        FROM {src},
        LATERAL FLATTEN(input => EXTRACTION:stock_items) f
        WHERE EXTRACTION:stock_items IS NOT NULL
    """
    _step = "BEGIN"
    cur.execute("BEGIN")
    try:
        _step = "DELETE stale rows"
        cur.execute(delete_sql)
        deleted = cur.rowcount
        _step = "INSERT patients"
        cur.execute(patient_insert_sql)
        patients = cur.rowcount
        _step = "INSERT visits"
        cur.execute(visit_insert_sql)
        visits = cur.rowcount
        if patients == 0 or visits == 0:
            raise AirflowFailException(
                f"INSERT produced 0 rows: patients={patients}, visits={visits}. "
                f"{src} had rows but none transferred — check column mapping between "
                f"EXTRACTED_FORMS and {dst}."
            )
        _step = "INSERT vitals"
        cur.execute(vitals_insert_sql)
        vitals = cur.rowcount
        _step = "INSERT doctor_notes"
        cur.execute(doctor_notes_insert_sql)
        doctor_notes = cur.rowcount
        inv_products = inv_stock_movements = 0
        if default_store:
            _step = "INSERT inv_products"
            cur.execute(inv_products_insert_sql)
            inv_products = cur.rowcount
            _step = "INSERT inv_stock_movements"
            cur.execute(inv_stock_movements_insert_sql)
            inv_stock_movements = cur.rowcount
        _step = "COMMIT"
        cur.execute("COMMIT")
    except Exception as exc:
        cur.execute("ROLLBACK")
        log.error("run_write rolled back at step '%s': %s", _step, exc)
        raise
    log.info(
        "Deleted %d stale row(s); inserted %d patient + %d visit + %d vital + %d doctor_note"
        " + %d inv_product + %d inv_stock_movement row(s) into %s",
        deleted, patients, visits, vitals, doctor_notes, inv_products, inv_stock_movements, dst,
    )
    return {"deleted": deleted, "patients": patients, "visits": visits,
            "vitals": vitals, "doctor_notes": doctor_notes,
            "inv_products": inv_products, "inv_stock_movements": inv_stock_movements}


def run_verify(cur, dst: str, facility: str) -> dict:
    """Confirm all four source_tables landed in EVENTS_RAW.

    Hard-fails if patients or visits are missing (pipeline cannot proceed without them).
    Logs vitals and doctor_notes counts as info — 0-row is acceptable if no clinical
    data was extracted, but non-zero confirms the evaluation INSERTs ran correctly.
    """
    reception_sql = f"""
        SELECT source_table, COUNT(*) AS n
        FROM {dst}
        WHERE facility_id = '{facility}'
          AND module_source = 'reception-service'
          AND source_table IN ('patients', 'visits')
        GROUP BY source_table
    """
    rows = cur.execute(reception_sql).fetchall()
    counts = {r[0]: int(r[1]) for r in rows}
    missing = [t for t in ("patients", "visits") if counts.get(t, 0) == 0]
    if missing:
        raise AirflowFailException(
            f"{dst} has no rows for source_table={missing} with "
            f"module_source='reception-service' after the write step. "
            f"The INSERT reported success but rows are not visible — "
            f"check Snowflake session isolation or transaction commit."
        )
    null_name_row = cur.execute(f"""
        SELECT COUNT_IF(payload:patient_name::STRING IS NULL), COUNT(*)
        FROM {dst}
        WHERE facility_id = '{facility}' AND source_table = 'patients'
    """).fetchone()
    if null_name_row and null_name_row[1] > 0 and null_name_row[0] == null_name_row[1]:
        raise AirflowFailException(
            f"All {null_name_row[1]} patient row(s) have NULL patient_name — "
            f"schema property names do not match run_write() field paths. "
            f"Check that the facility schema JSON uses 'patient_name' as the property name."
        )
    evaluation_sql = f"""
        SELECT source_table, COUNT(*) AS n
        FROM {dst}
        WHERE facility_id = '{facility}'
          AND module_source = 'patient-evaluation-service'
          AND source_table IN ('vitals', 'doctor_notes')
        GROUP BY source_table
    """
    eval_rows = cur.execute(evaluation_sql).fetchall()
    counts.update({r[0]: int(r[1]) for r in eval_rows})
    log.info(
        "%s: verified patients=%d visits=%d vitals=%d doctor_notes=%d staged for snowflake_to_v3_migration.",
        dst, counts["patients"], counts["visits"],
        counts.get("vitals", 0), counts.get("doctor_notes", 0),
    )
    inventory_sql = f"""
        SELECT source_table, COUNT(*) AS n
        FROM {dst}
        WHERE facility_id = '{facility}'
          AND module_source = 'inventory-service'
          AND source_table IN ('inv_products', 'inv_stock_movements')
        GROUP BY source_table
    """
    inv_rows = cur.execute(inventory_sql).fetchall()
    counts.update({r[0]: int(r[1]) for r in inv_rows})
    log.info(
        "%s: inventory — inv_products=%d inv_stock_movements=%d",
        dst, counts.get("inv_products", 0), counts.get("inv_stock_movements", 0),
    )
    return counts


@dag(
    dag_id=DAG_ID,
    description="Bridge: {facility}_DOCS.EXTRACTED_FORMS → {facility}_RAW.EVENTS_RAW (facility from conf)",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["snowflake", "documents", "migration"],
    default_args={"owner": "data-eng", "retries": 1, "retry_delay": timedelta(minutes=2)},
    params={
        "facility": Param("", type="string", title="Facility key", description='e.g. "siaya"'),
        "default_store": Param("", type="string", title="Default store UUID", description="UUID of the V3 store for opening stock. Required for stock-taking documents; leave blank for outpatient-only runs."),
    },
    on_success_callback=lap_callback,
    on_failure_callback=lap_callback,
)
def docs_to_events_raw():

    @task
    def validate() -> int:
        ctx = get_current_context()
        facility = ctx['dag_run'].conf.get('facility', '').strip().lower()
        _require_facility(facility)
        src = f"HOSPITALS.{facility.upper()}_DOCS.EXTRACTED_FORMS"
        dst = f"HOSPITALS.{facility.upper()}_RAW.EVENTS_RAW"
        use_pipelines_dir()
        import flatten_jsons_schemas as fl
        conn = fl._snowflake_connect()
        try:
            cur = conn.cursor()
            run_setup(cur, facility, dst)
            return run_validate(cur, src, facility)
        finally:
            conn.close()

    @task
    def write(row_count: int) -> dict:
        ctx = get_current_context()
        facility = ctx['dag_run'].conf.get('facility', '').strip().lower()
        _require_facility(facility)
        default_store = ctx['dag_run'].conf.get('default_store', '').strip()
        src = f"HOSPITALS.{facility.upper()}_DOCS.EXTRACTED_FORMS"
        dst = f"HOSPITALS.{facility.upper()}_RAW.EVENTS_RAW"
        use_pipelines_dir()
        import flatten_jsons_schemas as fl
        conn = fl._snowflake_connect()
        try:
            cur = conn.cursor()
            run_setup(cur, facility, dst)
            return run_write(cur, src, dst, facility, default_store)
        finally:
            conn.close()

    @task
    def verify(counts: dict) -> None:
        ctx = get_current_context()
        facility = ctx['dag_run'].conf.get('facility', '').strip().lower()
        _require_facility(facility)
        dst = f"HOSPITALS.{facility.upper()}_RAW.EVENTS_RAW"
        use_pipelines_dir()
        import flatten_jsons_schemas as fl
        conn = fl._snowflake_connect()
        try:
            run_verify(conn.cursor(), dst, facility)
        finally:
            conn.close()

    verify(write(validate()))


docs_to_events_raw()
