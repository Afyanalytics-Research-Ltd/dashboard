"""
sync_to_snowflake.py  —  Xana Pipeline
Write MySQL tables to Snowflake TENRI_RAW if they don't exist there or have 0 rows.
"""

import logging
from typing import List, Optional

import pandas as pd
from snowflake.connector.pandas_tools import write_pandas
from sqlalchemy import text
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)

SF_TARGET_SCHEMA = "TENRI_RAW"
SF_TARGET_DB     = "HOSPITALS"
CHUNK_SIZE       = 50_000


def _sf_row_count(sf_conn, table: str) -> int:
    """Return row count for table in Snowflake, or -1 if table doesn't exist."""
    try:
        cur = sf_conn.cursor()
        cur.execute(
            f"SELECT COUNT(*) FROM {SF_TARGET_DB}.{SF_TARGET_SCHEMA}.{table}"
        )
        return cur.fetchone()[0]
    except Exception:
        return -1


def _read_mysql_chunks(engine: Engine, schema: str, table: str):
    """Yield DataFrames in chunks from MySQL."""
    offset = 0
    while True:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(f"SELECT * FROM `{schema}`.`{table}` LIMIT {CHUNK_SIZE} OFFSET {offset}"),
                conn,
            )
        if df.empty:
            break
        yield df
        offset += CHUNK_SIZE


def sync_tables_to_snowflake(
    engine: Engine,
    mysql_schema: str,
    tables: List[str],
    sf_conn,
) -> None:
    """
    For each table in `tables`:
      - If it doesn't exist in Snowflake TENRI_RAW or has 0 rows, write all rows from MySQL.
      - Otherwise skip.
    """
    for table in tables:
        sf_count = _sf_row_count(sf_conn, table)
        if sf_count > 0:
            log.info("  SYNC   %-45s already has %d rows in Snowflake — skip", table, sf_count)
            continue

        action = "creating" if sf_count == -1 else "overwriting (0 rows)"
        log.info("  SYNC   %-45s %s in Snowflake ...", table, action)

        # Read all rows from MySQL
        total = 0
        first_chunk = True
        for df in _read_mysql_chunks(engine, mysql_schema, table):
            # Snowflake column names must be uppercase
            df.columns = [c.upper() for c in df.columns]

            # Cast datetime columns to string (pyarrow compatibility)
            dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            for col in dt_cols:
                df[col] = df[col].astype(str).replace("NaT", None)
            # Cast all object columns to string to prevent pyarrow mistyping
            # ICD10 codes (e.g. "K29.7") as FIXED/numeric when mixed with NULLs
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].apply(lambda x: None if pd.isnull(x) else str(x))

            success, nchunks, nrows, _ = write_pandas(
                conn           = sf_conn,
                df             = df,
                table_name     = table.upper(),
                database       = SF_TARGET_DB,
                schema         = SF_TARGET_SCHEMA,
                auto_create_table = first_chunk,   # create on first chunk only
                overwrite      = first_chunk,      # truncate+insert on first chunk
            )
            first_chunk = False
            total += nrows

        log.info("  SYNC   %-45s wrote %d rows to Snowflake", table, total)
