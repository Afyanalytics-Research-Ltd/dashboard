from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import snowflake.connector
import streamlit as st
from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parents[5]
load_dotenv(_PROJECT_ROOT / ".env")

_KEY_PATH = (_PROJECT_ROOT / os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "rsa_key.p8").strip()).resolve()


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


@st.cache_data(ttl=3600, show_spinner=False)
def run_query(sql: str) -> pd.DataFrame:
    """Execute SQL and return a DataFrame. Results cached for 1 hour."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(sql)
    cols = [d[0].upper() for d in cur.description]
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows, columns=cols)


def run_query_uncached(sql: str) -> pd.DataFrame:
    """Execute SQL without caching — use for write operations or real-time checks."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(sql)
    cols = [d[0].upper() for d in cur.description]
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows, columns=cols)
