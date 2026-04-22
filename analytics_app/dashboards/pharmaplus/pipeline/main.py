import sys
import os
import time
import logging
import polars as pl
from datetime import datetime
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent          # project root
DATA = Path(__file__).parent / "data"        # pipeline/data/
sys.path.insert(0, str(ROOT))
# ─────────────────────────────────────────────────────────────────────────────

from src.database import load_engine_inputs
from src.engine import apply_recommendation_logic, apply_bundle_logic, EngineConfig

# ----------------------------------------------------------------
# LOGGING CONFIGURATION
# ----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def run_analytics_pipeline(config: EngineConfig = None) -> pl.DataFrame | None:
    """
    PharmaPlus Kenya — Main Pipeline Orchestrator.

    Runs two independent engines:
        1. apply_recommendation_logic() → engine_output_latest.csv
        2. apply_bundle_logic()         → bundle_output_latest.csv

    Streamlit reads both CSVs independently.
    """
    if config is None:
        config = EngineConfig()

    # Timestamp defined once — both engines share it
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()

    logger.info("=" * 50)
    logger.info("🚀 PHARMA ANALYTICS ENGINE: STARTING PIPELINE")
    logger.info("=" * 50)

    # ----------------------------------------------------------------
    # 1. INGESTION
    # ----------------------------------------------------------------
    try:
        df_expiry, df_velocity, df_class, df_bundle_map = load_engine_inputs()
    except Exception as e:
        logger.error(f"❌ Pipeline Aborted: Database ingestion failed — {e}")
        return None

    DATA.mkdir(parents=True, exist_ok=True)
    df_expiry.write_csv(str(DATA / "expiry_stock_latest.csv"))

    # ----------------------------------------------------------------
    # 2. COLUMN VALIDATION
    # ----------------------------------------------------------------
    required = {
        "Expiry": [
            "snapshot_id", "product_id", "product_name", "facility_id",
            "qty_at_risk", "selling_price", "days_to_expiry",
            "discount_band_W", "ksh_value_at_risk"
        ],
        "Velocity": [
            "product_id", "facility_id",
            "daily_velocity", "days_to_clearance"
        ],
        "Classification": [
            "product_name", "bundle_eligible", "companion_link"
        ],
        "Bundle Map": [
            "expiring_product_id", "companion_product_id",
            "companion_daily_velocity", "companion_rank"
        ],
    }
    frames = {
        "Expiry": df_expiry, "Velocity": df_velocity,
        "Classification": df_class, "Bundle Map": df_bundle_map
    }
    validation_passed = True
    for name, cols in required.items():
        missing = [c for c in cols if c not in frames[name].columns]
        if missing:
            logger.error(f"❌ {name} DataFrame missing columns: {missing}")
            validation_passed = False

    if not validation_passed:
        logger.error("❌ Pipeline Aborted: Column validation failed.")
        return None

    # ----------------------------------------------------------------
    # 3. CORE ENGINE
    # ----------------------------------------------------------------
    logger.info("🧠 Running Recommendation Logic...")
    try:
        results = apply_recommendation_logic(df_expiry, df_velocity, df_class, config)
    except Exception as e:
        logger.error(f"❌ Core Engine Failed: {e}")
        return None

    if isinstance(results, pl.LazyFrame):
        results = results.collect()

    # ----------------------------------------------------------------
    # 4. BUNDLE ENGINE — runs independently, failure does not abort
    # ----------------------------------------------------------------
    logger.info("🔗 Running Bundle Logic...")
    try:
        bundle_results = apply_bundle_logic(df_bundle_map, df_expiry, config)

        bundle_results.write_csv(str(DATA / "bundle_output_latest.csv"))
        bundle_results.write_csv(str(DATA / f"bundle_output_{timestamp}.csv"))

        n_bundles  = bundle_results.filter(pl.col("companion_rank") == 1).shape[0]
        n_viable   = bundle_results.filter(
            (pl.col("companion_rank") == 1) & (pl.col("companion_viable") == True)
        ).shape[0]
        ksh_bundle = bundle_results.filter(pl.col("companion_rank") == 1)["bundle_recovered_value"].sum()
       

        logger.info("✅ BUNDLE ENGINE COMPLETE")
        logger.info(f"📁 Bundle output:              {DATA / 'bundle_output_latest.csv'}")
        logger.info("-" * 30)
        logger.info(f"🔗 Bundle pairs identified:    {n_bundles}")
        logger.info(f"✅ Viable pairs:               {n_viable}")
        logger.info(f"💰 KSh recoverable via bundle: KSh {ksh_bundle:,.2f}")

    except Exception as e:
        logger.error(f"❌ Bundle Engine Failed: {e}")

    # ----------------------------------------------------------------
    # 5. DIAGNOSTICS (remove before production)
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DIAGNOSTIC — DISCOUNT SIGNAL PER PRODUCT")
    print("=" * 60)
    print(
        results.select([
            "product_name", "days_to_expiry",
            "raw_discount_pct", "action"
        ]).sort("raw_discount_pct", descending=True)
    )
    print(
    results.select([
        "product_name", "clearance_prob", 
        "discount_recovery", "recovered_value","action"
    ]).filter(pl.col("action") == "DISCOUNT")
    )
    print(
    results.select([
        "product_name", "action",
        "bundle_eligible", "legal_status"
    ]).sort("product_name")
    )
    # ----------------------------------------------------------------
    # 6. EXPORT CORE ENGINE
    # ----------------------------------------------------------------
    results.write_csv(str(DATA / f"engine_output_{timestamp}.csv"))
    results.write_csv(str(DATA / "engine_output_latest.csv"))

    # ----------------------------------------------------------------
    # 7. SUMMARY REPORT (Updated Terminology)
    # ----------------------------------------------------------------
    total_at_risk   = results["ksh_value_at_risk"].sum()
    total_recovered = results["recovered_value"].sum()
    
    logger.info("=" * 50)
    logger.info("✅ PIPELINE COMPLETE: REVENUE MITIGATION STRATEGY")
    logger.info("-" * 30)
    logger.info(f"🚨 Total Revenue at Risk:  KSh {total_at_risk:,.0f}")
    logger.info(f"🛡️  Value Recovery:         KSh {total_recovered:,.0f}")
    
    # New check for Bundling
    if 'bundle_results' in locals():
        viable_bundles = bundle_results.filter(pl.col("companion_viable") == True)
        bundle_rev = viable_bundles["projected_bundle_revenue"].sum()
        logger.info(f"💰 Recoverable via Bundle: KSh {bundle_rev:,.0f}")
    
    logger.info("=" * 50)

    return results


if __name__ == "__main__":
    run_analytics_pipeline()