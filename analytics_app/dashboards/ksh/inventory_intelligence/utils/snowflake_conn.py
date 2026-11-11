from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import snowflake.connector
import streamlit as st
from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parents[5]
load_dotenv(_PROJECT_ROOT / ".env")

_KEY_PATH = (_PROJECT_ROOT / os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "rsa_key.p8").strip()).resolve()

logger = logging.getLogger(__name__)

# 390114: "Authentication token has expired. The user must authenticate
# again." Raised once the connector's own internal session-renewal (using
# the master token) has itself failed — the cached connection has been idle
# long enough that both the session token and the master token that would
# normally refresh it are gone. This container runs as one long-lived
# st.cache_resource connection shared across every rerun, so a fresh login
# (not a renewal) is needed; see snowflake_client.py for the same fix
# elsewhere in this repo.
_TOKEN_EXPIRED_ERRNO = 390114


@st.cache_resource(show_spinner=False)
def _get_connection() -> snowflake.connector.SnowflakeConnection:
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT").strip(),
        user=os.getenv("SNOWFLAKE_USER").strip(),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE").strip(),
        database=os.getenv("SNOWFLAKE_DATABASE").strip(),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC").strip(),
        private_key_file=str(_KEY_PATH),
    )


def _execute(sql: str) -> pd.DataFrame:
    conn = _get_connection()
    cur = conn.cursor()
    try:
        cur.execute(sql)
        cols = [d[0].upper() for d in cur.description]
        rows = cur.fetchall()
    finally:
        cur.close()
    return pd.DataFrame(rows, columns=cols)


def _execute_with_retry(sql: str) -> pd.DataFrame:
    try:
        return _execute(sql)
    except snowflake.connector.errors.ProgrammingError as exc:
        if getattr(exc, "errno", None) != _TOKEN_EXPIRED_ERRNO:
            raise
        logger.warning("Snowflake session token expired — reconnecting and retrying once.")
        _get_connection.clear()
        return _execute(sql)


@st.cache_data(ttl=3600, show_spinner=False)
def run_query(sql: str) -> pd.DataFrame:
    """Execute SQL and return a DataFrame. Results cached for 1 hour."""
    return _execute_with_retry(sql)


def run_query_uncached(sql: str) -> pd.DataFrame:
    """Execute SQL without caching — use for write operations or real-time checks."""
    return _execute_with_retry(sql)
