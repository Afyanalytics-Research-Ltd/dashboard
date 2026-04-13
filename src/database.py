import os
import logging
import polars as pl
from dotenv import load_dotenv

load_dotenv()


def get_db_url() -> str:
    """
    Builds MySQL connection URI from environment variables.
    ConnectorX requires plain mysql:// — no +pymysql driver suffix.
    """
    user     = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host     = os.getenv("DB_HOST")
    port     = os.getenv("DB_PORT")
    name     = os.getenv("DB_NAME")

    if not all([user, password, host, port, name]):
        raise EnvironmentError(
            "❌ Missing database credentials. "
            "Check DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME in .env"
        )

    return f"mysql://{user}:{password}@{host}:{port}/{name}"


def fetch_data(query: str) -> pl.DataFrame:
    """
    Executes a SQL query and returns a Polars DataFrame.
    Uses ConnectorX engine for maximum ingestion speed.
    """
    uri = get_db_url()
    try:
        return pl.read_database_uri(query=query, uri=uri, engine="connectorx")
    except Exception as e:
        logging.error(f"❌ Data Ingestion Error: {e}")
        raise


def load_engine_inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Loads and normalizes all engine input DataFrames.

    Returns:
        df_expiry    : v_expiry_stock — expiring products per store
        df_velocity  : v_sales_velocity — daily velocity per product × facility
        df_class     : inventory_product_classifications_v2 — bundle/legal metadata
        df_bundle_map: tmp_bundle_map — top 3 velocity companions per expiring product
    """

    # ----------------------------------------------------------------
    # 1. Expiry stock
    # Normalize product_name for reliable joins to classifications
    # ----------------------------------------------------------------
    df_expiry = fetch_data("SELECT * FROM terra.v_expiry_stock").with_columns([
        pl.col("product_name").str.strip_chars().str.to_uppercase()
    ])
    logging.info(f"✅ Expiry stock loaded:      {df_expiry.shape[0]} rows")

    # ----------------------------------------------------------------
    # 2. Sales velocity
    # No normalization needed — product_id join is numeric
    # ----------------------------------------------------------------
    df_velocity = fetch_data("SELECT * FROM terra.v_sales_velocity")
    logging.info(f"✅ Sales velocity loaded:    {df_velocity.shape[0]} rows")

    # ----------------------------------------------------------------
    # 3. Product classifications
    # companion_link: AI-generated CSV used string "NONE", "N/A", ""
    # as placeholders instead of NULL. Convert all to actual null so
    # the bundle companion join correctly finds no match rather than
    # looking for a product literally named "NONE".
    # ----------------------------------------------------------------
    df_class = fetch_data(
        "SELECT * FROM terra.inventory_product_classifications_v2"
    ).with_columns([
        pl.col("product_name").str.strip_chars().str.to_uppercase(),
        pl.col("bundle_eligible").cast(pl.Boolean),
        pl.when(
            pl.col("companion_link").is_null() |
            pl.col("companion_link").str.strip_chars().str.to_uppercase()
              .is_in(["NONE", "N/A", "NULL", ""])
        )
        .then(pl.lit(None))
        .otherwise(
            pl.col("companion_link").str.strip_chars().str.to_uppercase()
        )
        .alias("companion_link")
    ])
    logging.info(f"✅ Classifications loaded:   {df_class.shape[0]} rows")

    # ----------------------------------------------------------------
    # 4. Bundle map
    # Top 3 velocity-ranked OTC companions per expiring product × facility
    # Sourced from tmp_bundle_map — materialized table, not a view
    # ----------------------------------------------------------------
    df_bundle_map = fetch_data("""
        SELECT *
        FROM terra.tmp_bundle_map
        WHERE companion_rank <= 3
    """)
    logging.info(f"✅ Bundle map loaded:        {df_bundle_map.shape[0]} rows")

    return df_expiry, df_velocity, df_class, df_bundle_map
















   