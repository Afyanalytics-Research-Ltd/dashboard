import os
import logging

import pandas as pd
import snowflake.connector

logger = logging.getLogger(__name__)


def run_query_df(sql: str) -> pd.DataFrame:
    """Execute a SELECT against Snowflake, return a DataFrame.

    Reads connection params from env vars — same vars used by the main
    dashboard (SNOWFLAKE_USER, SNOWFLAKE_ACCOUNT, SNOWFLAKE_WAREHOUSE,
    SNOWFLAKE_DATABASE, SNOWFLAKE_PRIVATE_KEY_PATH or SNOWFLAKE_PASSWORD).
    Opens and closes a connection per call — suitable for scheduled jobs,
    not for per-request use.
    """
    conn_kwargs = dict(
        user=os.getenv("SNOWFLAKE_USER", "").strip(),
        account=os.getenv("SNOWFLAKE_ACCOUNT", "").strip(),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "").strip(),
        database=os.getenv("SNOWFLAKE_DATABASE", "").strip(),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC").strip(),
    )
    private_key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "").strip()
    if private_key_path:
        conn_kwargs["private_key_file"] = private_key_path
    else:
        conn_kwargs["password"] = os.getenv("SNOWFLAKE_PASSWORD", "").strip()

    conn = snowflake.connector.connect(**conn_kwargs)
    try:
        cur = conn.cursor()
        try:
            cur.execute(sql)
            df = cur.fetch_pandas_all()
            logger.debug("snowflake_db: %d rows returned", len(df))
            return df
        finally:
            cur.close()
    except Exception as exc:
        logger.error("snowflake_db query failed: %s", exc)
        raise
    finally:
        conn.close()
