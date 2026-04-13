"""
xana_build_diagnosis_mart.py  —  Xana Pipeline
══════════════════════════════════════════════════════════════════════════════
Step 4: Build fact_visit_diagnoses in HOSPITALS.REPORTING.

READS FROM
──────────
    HOSPITALS.STAGING.stg_visits                — visit context + facility_key
    HOSPITALS.STAGING.stg_icd10                 — structured ICD-10 per visit
    HOSPITALS.REPORTING.REPORTING_fact_doctor_notes_cleaned — NLP output (Step 3)

THE MERGE PROBLEM (and how we solve it)
──────────────────────────────────────────
The original Tendri mart joined stg_unified_icd10 to REPORTING_fact_doctor_notes_cleaned
by sk_visit_id. The problem: these two tables had different grains.

  stg_unified_icd10   — one row per ICD-10 code per visit
                         (a visit with 3 diagnoses = 3 rows)
  fact_doctor_notes   — one row per visit

Joining them directly caused fan-out: each NLP row was duplicated for every
ICD-10 row on the same visit, inflating the mart. The original code
used a window function + pivot (ROW_NUMBER PARTITION BY sk_visit_id) to
collapse stg_unified_icd10 to one row per visit before joining. That
window scan was expensive.

HERE we avoid the problem entirely because:

  stg_icd10           — already one row per visit (primary code only,
                         resolved at ingest time in evaluation_doctor_notes_merged
                         via ROW_NUMBER at merge time, not query time)

So the join is:
    stg_visits (1 row/visit)
    LEFT JOIN stg_icd10 (1 row/visit)          ← no fan-out
    LEFT JOIN REPORTING_fact_doctor_notes_cleaned (1 row/visit) ← no fan-out

Result: clean 1:1:1 join, no window functions at mart query time.

WRITES TO
──────────
    HOSPITALS.REPORTING.fact_visit_diagnoses

USAGE
──────
    python xana_build_diagnosis_mart.py          # incremental
    python xana_build_diagnosis_mart.py --full   # full rebuild
══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import logging
from datetime import datetime

import pandas as pd
from snowflake.connector.pandas_tools import write_pandas
from sqlalchemy import text
from sqlalchemy.engine import Engine

from config import STAGING_SCHEMA, REPORTING_SCHEMA
from db import build_mysql_engine

SF_TARGET_DB     = "HOSPITALS"
SF_TARGET_SCHEMA = "TENRI_RAW"

log = logging.getLogger(__name__)

TARGET_TABLE   = "fact_visit_diagnoses"
TARGET_SCHEMA  = REPORTING_SCHEMA



# ══════════════════════════════════════════════════════════════════════════════
#  LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_mysql(engine: Engine, table: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text(f"SELECT * FROM `{STAGING_SCHEMA}`.`{table}`"), conn
        )


def _load_nlp_from_sf(sf_conn) -> pd.DataFrame:
    """Read REPORTING_fact_doctor_notes_cleaned from Snowflake."""
    try:
        cur = sf_conn.cursor()
        cur.execute(
            f"SELECT * FROM {SF_TARGET_DB}.{SF_TARGET_SCHEMA}"
            f".FACT_DOCTOR_NOTES_CLEANED"
        )
        cols = [d[0].lower() for d in cur.description]
        rows = cur.fetchall()
        cur.close()
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        log.warning("  NLP table not found in Snowflake — NLP columns will be NULL")
        return pd.DataFrame()


def _sf_existing_mart_ids(sf_conn) -> set:
    try:
        cur = sf_conn.cursor()
        cur.execute(
            f"SELECT DISTINCT SK_VISIT_ID "
            f"FROM {SF_TARGET_DB}.{SF_TARGET_SCHEMA}.{TARGET_TABLE.upper()}"
        )
        ids = {row[0] for row in cur.fetchall()}
        cur.close()
        return ids
    except Exception:
        return set()


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD MART DATAFRAME  (pandas merge — no MySQL cross-table JOIN needed)
# ══════════════════════════════════════════════════════════════════════════════

def _build_mart_df(
    stg_visits: pd.DataFrame,
    stg_icd10: pd.DataFrame,
    nlp: pd.DataFrame,
) -> pd.DataFrame:
    df = stg_visits.merge(
        stg_icd10[[
            "sk_visit_id", "icd10_note_code", "icd10_variation_code",
            "icd10_variation_name", "icd10_type_code", "icd10_type_name",
            "icd10_subcategory_name", "icd10_category_name",
        ]],
        on="sk_visit_id", how="left",
    )

    if not nlp.empty:
        nlp_cols = [
            "visit_id", "primary_disease", "comorbidity_1", "comorbidity_2",
            "coalesced_diagnosis", "nlp_primary_icd10_code", "is_chronic",
            "clinical_status", "nlp_cleaned_diagnosis", "is_rule_out", "is_invalid",
        ]
        nlp_sub = nlp[[c for c in nlp_cols if c in nlp.columns]].copy()
        df = df.merge(nlp_sub, left_on="sk_visit_id", right_on="visit_id", how="left")
    else:
        for col in ["primary_disease", "comorbidity_1", "comorbidity_2",
                    "coalesced_diagnosis", "nlp_primary_icd10_code", "is_chronic",
                    "clinical_status", "nlp_cleaned_diagnosis", "is_rule_out", "is_invalid"]:
            df[col] = None

    df["combined_primary_disease"] = df["icd10_variation_name"].where(
        df["icd10_variation_name"].notna(), df.get("primary_disease")
    )
    df["has_icd10"]       = df["icd10_note_code"].notna().astype(int)
    df["has_nlp"]         = df.get("visit_id", pd.Series(dtype=object)).notna().astype(int)
    df["is_invalid_note"] = df.get("is_invalid", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["is_chronic"]      = df.get("is_chronic", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["refreshed_at"]    = datetime.now()

    keep = [
        "sk_visit_id", "source_schema", "sk_patient_id", "facility_key",
        "visit_start_date", "payment_mode", "type_of_visit", "visit_type",
        "icd10_note_code", "icd10_variation_code", "icd10_variation_name",
        "icd10_type_code", "icd10_type_name", "icd10_subcategory_name", "icd10_category_name",
        "primary_disease", "comorbidity_1", "comorbidity_2", "coalesced_diagnosis",
        "nlp_primary_icd10_code", "is_chronic", "clinical_status",
        "nlp_cleaned_diagnosis", "is_rule_out",
        "combined_primary_disease", "has_icd10", "has_nlp",
        "is_invalid_note", "refreshed_at",
    ]
    # Rename NLP columns to match DDL
    rename = {
        "primary_disease": "nlp_primary_disease",
        "comorbidity_1":   "nlp_comorbidity_1",
        "comorbidity_2":   "nlp_comorbidity_2",
        "coalesced_diagnosis": "nlp_coalesced_diagnosis",
        "is_chronic":      "nlp_is_chronic",
        "clinical_status": "nlp_clinical_status",
        "nlp_cleaned_diagnosis": "nlp_cleaned_diagnosis",
    }
    df = df.rename(columns=rename)
    final_cols = [rename.get(c, c) for c in keep]
    present = [c for c in final_cols if c in df.columns]
    return df[present].drop_duplicates(subset=["sk_visit_id"])


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def refresh_mart(engine: Engine, full: bool = False, sf_conn=None) -> None:
    """
    Entry point called from run_pipeline.py Step 4.
    Reads stg_visits + stg_icd10 from MySQL, NLP table from Snowflake,
    merges in pandas, writes result to Snowflake TENRI_RAW.
    """
    log.info("=" * 60)
    log.info("Step 4: Diagnosis mart (%s)", "FULL" if full else "INCREMENTAL")
    log.info("=" * 60)

    stg_visits = _load_mysql(engine, "stg_visits")
    stg_icd10  = _load_mysql(engine, "stg_icd10")
    nlp        = _load_nlp_from_sf(sf_conn) if sf_conn else pd.DataFrame()

    has_icd10_ids = set(stg_icd10["sk_visit_id"])
    has_nlp_ids   = set(nlp["visit_id"]) if not nlp.empty and "visit_id" in nlp.columns else set()
    candidate_ids = has_icd10_ids | has_nlp_ids

    stg_visits = stg_visits[stg_visits["sk_visit_id"].isin(candidate_ids)]

    if not full and sf_conn:
        existing_ids = _sf_existing_mart_ids(sf_conn)
        stg_visits = stg_visits[~stg_visits["sk_visit_id"].isin(existing_ids)]

    if stg_visits.empty:
        log.info("  Nothing to process — mart is up to date")
        return

    log.info("  Visits to process: %d", len(stg_visits))
    mart_df = _build_mart_df(stg_visits, stg_icd10, nlp)

    if sf_conn and not mart_df.empty:
        df_sf = mart_df.copy()
        df_sf.columns = [c.upper() for c in df_sf.columns]
        dt_cols = [c for c in df_sf.columns if pd.api.types.is_datetime64_any_dtype(df_sf[c])]
        for col in dt_cols:
            df_sf[col] = df_sf[col].astype(str).replace("NaT", None)
        for col in df_sf.select_dtypes(include=["object"]).columns:
            df_sf[col] = df_sf[col].apply(lambda x: None if pd.isnull(x) else str(x))
        log.info("  [DEBUG Step4] dtypes before write_pandas:\n%s", df_sf.dtypes.to_string())
        write_pandas(
            conn=sf_conn,
            df=df_sf,
            table_name=TARGET_TABLE.upper(),
            database=SF_TARGET_DB,
            schema=SF_TARGET_SCHEMA,
            auto_create_table=True,
            overwrite=full,
        )
        log.info("  Wrote %d rows to Snowflake %s.%s", len(mart_df), SF_TARGET_SCHEMA, TARGET_TABLE)

    log.info("=" * 60)
    log.info("Step 4 complete")
    log.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Xana: Visit diagnoses mart")
    parser.add_argument("--full", action="store_true", help="Full rebuild")
    args = parser.parse_args()
    engine = build_mysql_engine()
    refresh_mart(engine, full=args.full)