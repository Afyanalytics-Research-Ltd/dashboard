"""
PharmaPlus Kenya — Data Loader
================================
Single entry point for all engine inputs.

  load_demo_inputs()    — CSV mode (demo)
  load_engine_inputs()  — MySQL mode (production)

Both return an identical tuple:
    (df_expiry, df_geo, df_serp)

    df_expiry : expiry stock with seasonal_boost already applied
    df_geo    : branch market DNA with facility_id attached
    df_serp   : competitor price comparison table (or None)

The engine and app receive clean, ready-to-use DataFrames and
know nothing about file paths or data sources.
"""

import logging
import os
from datetime import date
from typing import Optional

import polars as pl
from dotenv import load_dotenv

load_dotenv()

# ── Column contracts ───────────────────────────────────────────────────────────
EXPIRY_REQUIRED = [
    "product_id", "product_name", "facility_id", "store_name",
    "segment", "days_to_expiry", "expiry_tier",
    "qty_at_risk", "selling_price", "unit_cost",
    "ksh_value_at_risk", "daily_velocity",
]

GEO_REQUIRED = [
    "Branch", "Est_Pop_2026",
    "Lifestyle_Drivers", "Social_Drivers",
    "Competition", "High_Pressure_Comp",
]

BRANCH_FACILITY_MAP = {
    "Branch Yaya":       1,
    "Branch Westlands":  2,
    "Branch Imara Mall": 3,
}


# ── Demo loader ────────────────────────────────────────────────────────────────
def load_demo_inputs(
    expiry_path: str = "data/expiry_stock_sim.csv",
    geo_path:    str = "data/branch_market_dna.csv",
    serp_path:   str = "data/competitor_prices.csv",
    ref_date:    Optional[date] = None,
) -> tuple:
    """
    Load all engine inputs from CSV files (demo mode).
    Applies seasonal signals before returning.

    Returns:
        (df_expiry, df_geo, df_serp)
    """
    if not os.path.exists(expiry_path):
        raise FileNotFoundError(
            f"Demo expiry data not found at '{expiry_path}'. "
            "Run src/simulate_data.py first."
        )

    df_expiry = (
        pl.read_csv(expiry_path)
        .with_columns([
            pl.col("daily_velocity").cast(pl.Float64),
            pl.col("qty_at_risk").cast(pl.Float64),
            pl.col("selling_price").cast(pl.Float64),
            pl.col("unit_cost").cast(pl.Float64),
            pl.col("ksh_value_at_risk").cast(pl.Float64),
            pl.col("days_to_expiry").cast(pl.Float64),
        ])
    )

    _validate_required(df_expiry, EXPIRY_REQUIRED, "df_expiry")

    # Apply seasonal signals
    df_expiry = _apply_seasonal(df_expiry, ref_date)
    logging.info(f"Demo expiry loaded: {df_expiry.shape[0]} rows")

    df_geo  = _load_geo(geo_path)
    df_serp = _load_serp(serp_path)

    return df_expiry, df_geo, df_serp


# ── Production loader ──────────────────────────────────────────────────────────
def load_engine_inputs(
    geo_path:  str = "data/branch_market_dna.csv",
    serp_path: str = "data/competitor_prices.csv",
    ref_date:  Optional[date] = None,
) -> tuple:
    """
    Load all engine inputs from production MySQL (terra database).
    Requires DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME in .env

    Returns:
        (df_expiry, df_geo, df_serp)
    """
    from src.database import fetch_data

    df_expiry = (
        fetch_data("SELECT * FROM terra.v_expiry_stock")
        .with_columns([
            pl.col("product_name").str.strip_chars().str.to_uppercase(),
        ])
    )

    if "seasonal_boost" not in df_expiry.columns:
        df_expiry = df_expiry.with_columns(pl.lit(1.0).alias("seasonal_boost"))

    _validate_required(df_expiry, EXPIRY_REQUIRED, "df_expiry (production)")

    df_expiry = _apply_seasonal(df_expiry, ref_date)
    logging.info(f"Production expiry loaded: {df_expiry.shape[0]} rows")

    df_geo  = _load_geo(geo_path)
    df_serp = _load_serp(serp_path)

    return df_expiry, df_geo, df_serp


# ── Seasonal helper ────────────────────────────────────────────────────────────
def _apply_seasonal(
    df: pl.DataFrame,
    ref_date: Optional[date] = None,
) -> pl.DataFrame:
    """
    Apply seasonal_flag and seasonal_boost to df_expiry.
    Overwrites any placeholder values already in the DataFrame.
    """
    from src.seasonal import apply_seasonal_signals
    return apply_seasonal_signals(df, today=ref_date or date.today())


# ── Geo helper ─────────────────────────────────────────────────────────────────
def _load_geo(geo_path: str) -> Optional[pl.DataFrame]:
    """
    Load branch_market_dna.csv and attach facility_id.
    Returns None if file not found — geo signals are optional.
    """
    if not os.path.exists(geo_path):
        logging.warning(f"Geo DNA not found at '{geo_path}' — geo enrichment skipped.")
        return None

    df_geo = pl.read_csv(geo_path)
    _validate_required(df_geo, GEO_REQUIRED, "df_geo")

    df_geo = df_geo.with_columns([
        pl.col("Branch")
        .replace(BRANCH_FACILITY_MAP)
        .cast(pl.Int64)
        .alias("facility_id")
    ])

    logging.info(f"Geo DNA loaded: {df_geo.shape[0]} branches")
    return df_geo


# ── Competitor pricing helper ──────────────────────────────────────────────────
def _load_serp(serp_path: str) -> Optional[pl.DataFrame]:
    """
    Load competitor_prices.csv. If not found, runs build_competitor_prices()
    to generate it. Returns None if generation also fails.
    """
    if not os.path.exists(serp_path):
        logging.info(
            f"competitor_prices.csv not found at '{serp_path}' — "
            "running build_competitor_prices()..."
        )
        try:
            from src.competitor_pricing import build_competitor_prices
            return build_competitor_prices(output_path=serp_path)
        except Exception as e:
            logging.warning(f"Could not build competitor prices: {e}")
            return None

    df_serp = pl.read_csv(serp_path)
    logging.info(f"Competitor prices loaded: {df_serp.shape[0]} SKUs")
    return df_serp


# ── Validation helper ──────────────────────────────────────────────────────────
def _validate_required(df: pl.DataFrame, required: list, name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{name}] Missing required columns: {missing}\n"
            f"Available: {df.columns}"
        )
