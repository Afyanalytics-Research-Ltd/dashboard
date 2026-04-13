"""
run_pipeline.py  —  Xana Snowflake Pipeline
══════════════════════════════════════════════════════════════════════════════
Pipeline orchestrator. Runs all steps in order.

STAGES
──────
    Step 1   snowflake_ingest    Snowflake → xana_raw (build_ingest.py)
    Step 2   staging             xana_raw → xana_staging (build_staging.py)
    Step 3   nlp_cleaning        NLP doctor notes cleaning (build_doctor_diagnosis_cleaned_v2.py)
    Step 4   diagnosis_mart      Visit diagnoses mart (build_visit_diagnoses_mart.py)

USAGE
──────
    # Incremental (day-to-day)
    python tendri/run_pipeline.py

    # First run or full rebuild
    python tendri/run_pipeline.py --full

    # Only run Snowflake ingest
    python tendri/run_pipeline.py --stop-after ingest

    # Skip Snowflake ingest (raw tables already current)
    python tendri/run_pipeline.py --skip-ingest

    # Resume after failure
    python tendri/run_pipeline.py --resume

    # Provide Snowflake MFA code
    python tendri/run_pipeline.py --totp 123456
══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime

from sqlalchemy import text

from config import REPORTING_DB, STAGING_DB, RAW_DB
from db import build_mysql_engine

log = logging.getLogger(__name__)

STAGES = ["snowflake_ingest", "staging", "nlp_cleaning", "diagnosis_mart"]


def banner(msg: str) -> None:
    width = 60
    log.info("=" * width)
    log.info(msg)
    log.info("=" * width)


# ── Run log table ─────────────────────────────────────────────────────────────

_RUN_LOG_DDL = f"""
CREATE TABLE IF NOT EXISTS `{STAGING_DB}`.`pipeline_run_log` (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    started_at      DATETIME NOT NULL,
    finished_at     DATETIME,
    status          VARCHAR(20),
    last_stage      VARCHAR(50),
    stages_done     TEXT,
    error_message   TEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

_STAGE_LOG_DDL = f"""
CREATE TABLE IF NOT EXISTS `{STAGING_DB}`.`pipeline_stage_log` (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    run_id      INT NOT NULL,
    stage       VARCHAR(50),
    started_at  DATETIME,
    finished_at DATETIME,
    status      VARCHAR(20),
    message     TEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def _ensure_log_tables(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(
            f"CREATE DATABASE IF NOT EXISTS `{STAGING_DB}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        ))
        conn.execute(text(_RUN_LOG_DDL))
        conn.execute(text(_STAGE_LOG_DDL))


def _start_run(engine) -> int:
    with engine.begin() as conn:
        result = conn.execute(text(
            f"INSERT INTO `{STAGING_DB}`.`pipeline_run_log` "
            "(started_at, status) VALUES (NOW(), 'running')"
        ))
        return result.lastrowid


def _finish_run(engine, run_id: int, status: str, error: str = "") -> None:
    with engine.begin() as conn:
        conn.execute(text(
            f"UPDATE `{STAGING_DB}`.`pipeline_run_log` "
            "SET finished_at=NOW(), status=:s, error_message=:e "
            "WHERE id=:id"
        ), {"s": status, "e": error[:2000], "id": run_id})


def _start_stage(engine, run_id: int, stage: str) -> int:
    with engine.begin() as conn:
        result = conn.execute(text(
            f"INSERT INTO `{STAGING_DB}`.`pipeline_stage_log` "
            "(run_id, stage, started_at, status) VALUES (:r, :s, NOW(), 'running')"
        ), {"r": run_id, "s": stage})
        return result.lastrowid


def _finish_stage(engine, stage_id: int, status: str, message: str = "") -> None:
    with engine.begin() as conn:
        conn.execute(text(
            f"UPDATE `{STAGING_DB}`.`pipeline_stage_log` "
            "SET finished_at=NOW(), status=:s, message=:m WHERE id=:id"
        ), {"s": status, "m": message[:2000], "id": stage_id})


def _last_run_completed_stages(engine) -> set:
    """Return stages that completed successfully in the most recent failed run."""
    try:
        with engine.connect() as conn:
            run_id = conn.execute(text(
                f"SELECT id FROM `{STAGING_DB}`.`pipeline_run_log` "
                "WHERE status != 'success' ORDER BY id DESC LIMIT 1"
            )).scalar()
            if not run_id:
                return set()
            rows = conn.execute(text(
                f"SELECT stage FROM `{STAGING_DB}`.`pipeline_stage_log` "
                "WHERE run_id=:r AND status='success'"
            ), {"r": run_id}).fetchall()
            return {r[0] for r in rows}
    except Exception:
        return set()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline(args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("xana_pipeline.log"),
        ],
    )

    engine = build_mysql_engine()
    _ensure_log_tables(engine)

    skip_stages: set = set()
    if args.resume:
        skip_stages = _last_run_completed_stages(engine)
        log.info("Resume mode: skipping completed stages: %s",
                 skip_stages or "none")

    run_id = _start_run(engine)
    started = datetime.now()
    log.info("Run ID: %d", run_id)

    full = args.full
    totp = getattr(args, "totp", None)

    # Resolve TOTP once — reused for ingest + sync
    if not totp and not os.getenv("SF_TOKEN"):
        import configparser as _cp
        _cfg = _cp.ConfigParser()
        _cfg.read(os.path.join(os.path.dirname(__file__), "..", "config.ini"))
        if not _cfg.get("snowflake", "token", fallback="").strip():
            totp = input("Enter your Duo TOTP code: ").strip() or None

    from db import build_snowflake_conn
    from sync_to_snowflake import sync_tables_to_snowflake
    sf_conn = build_snowflake_conn(totp=totp)

    try:
        # ── Step 1: Snowflake ingest ──────────────────────────────────────────
        if not args.skip_ingest and "snowflake_ingest" not in skip_stages:
            banner("STEP 1: Snowflake ingest")
            sid = _start_stage(engine, run_id, "snowflake_ingest")
            try:
                from build_ingest import run_ingest
                run_ingest(engine, full=full, sf_conn=sf_conn)
                _finish_stage(engine, sid, "success")
            except ImportError:
                log.warning("build_ingest.py not found — skipping")
                _finish_stage(engine, sid, "failed", "ImportError")
            except Exception:
                err = traceback.format_exc()
                _finish_stage(engine, sid, "failed", err[-1000:])
                _finish_run(engine, run_id, "failed", err[-500:])
                raise
        else:
            log.info("STEP 1: Skipping snowflake_ingest")

        if getattr(args, "stop_after", None) == "ingest":
            _finish_run(engine, run_id, "success")
            return

        # ── Step 2: Staging ───────────────────────────────────────────────────
        if "staging" not in skip_stages:
            banner("STEP 2: Staging tables")
            sid = _start_stage(engine, run_id, "staging")
            try:
                from building_stage import run_staging
                run_staging(engine, full=full)
                _finish_stage(engine, sid, "success")
            except Exception:
                err = traceback.format_exc()
                _finish_stage(engine, sid, "failed", err[-1000:])
                _finish_run(engine, run_id, "failed", err[-500:])
                raise
        else:
            log.info("STEP 2: Skipping staging")

        if getattr(args, "stop_after", None) == "staging":
            _finish_run(engine, run_id, "success")
            return

        # ── Step 3: NLP cleaning ──────────────────────────────────────────────
        if "nlp_cleaning" not in skip_stages:
            banner("STEP 3: NLP diagnosis cleaning")
            sid = _start_stage(engine, run_id, "nlp_cleaning")
            try:
                from build_nlp_cleaning import refresh_nlp
                refresh_nlp(engine, full=full, sf_conn=sf_conn)
                _finish_stage(engine, sid, "success")
            except Exception:
                err = traceback.format_exc()
                _finish_stage(engine, sid, "failed", err[-1000:])
                _finish_run(engine, run_id, "failed", err[-500:])
                raise
        else:
            log.info("STEP 3: Skipping nlp_cleaning")

        # ── Step 4: Diagnosis mart ────────────────────────────────────────────
        if "diagnosis_mart" not in skip_stages and not getattr(args, "skip_mart", False):
            banner("STEP 4: Diagnosis mart")
            sid = _start_stage(engine, run_id, "diagnosis_mart")
            try:
                from build_diagnosis_mart import refresh_mart
                refresh_mart(engine, full=full, sf_conn=sf_conn)
                _finish_stage(engine, sid, "success")
            except ImportError:
                log.warning("build_diagnosis_mart.py not found — skipping")
                _finish_stage(engine, sid, "success", "file not found")
            except Exception:
                err = traceback.format_exc()
                _finish_stage(engine, sid, "failed", err[-1000:])
                _finish_run(engine, run_id, "failed", err[-500:])
                raise
        else:
            log.info("STEP 4: Skipping diagnosis_mart")

        # ── Sync: all processed tables → Snowflake (only if 0 rows there) ────
        if sf_conn is not None:
            banner("STEP 5: Sync MySQL → Snowflake (empty tables only)")
            sync_tables_to_snowflake(
                engine, RAW_DB,
                [
                    # raw tables (originally pulled from Snowflake)
                    "reception_patients",
                    "evaluation_visits",
                    "evaluation_doctor_notes",
                    "evaluation_icd10_notes",
                    "evaluation_icd10_variations",
                    "evaluation_icd10_types",
                    "evaluation_icd10_subcategories",
                    "evaluation_icd10_categories",
                    "evaluation_prescriptions",
                    "finance_invoices",
                    # merged (built in Step 1)
                    "evaluation_doctor_notes_merged",
                    # reporting (built in Steps 3 & 4)
                    "fact_doctor_notes_cleaned",
                    "fact_visit_diagnoses",
                ],
                sf_conn,
            )

    except Exception:
        log.error("Pipeline failed — check xana_pipeline.log for details")
        raise
    else:
        elapsed = (datetime.now() - started).total_seconds()
        _finish_run(engine, run_id, "success")
        banner(f"PIPELINE COMPLETE — {elapsed:.1f}s  (run_id={run_id})")
        log.info("Power BI: connect to schema %s", STAGING_DB)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xana Snowflake Pipeline")
    parser.add_argument("--full",         action="store_true", help="Full rebuild all tables")
    parser.add_argument("--resume",       action="store_true", help="Skip stages completed in last failed run")
    parser.add_argument("--skip-ingest",  action="store_true", help="Skip Step 1 Snowflake ingest")
    parser.add_argument("--skip-mart",    action="store_true", help="Skip Step 4 diagnosis mart")
    parser.add_argument("--stop-after",   default=None,        choices=["ingest", "staging"],
                        help="Stop after this stage")
    parser.add_argument("--totp",         default=None,        help="Snowflake MFA passcode")
    run_pipeline(parser.parse_args())