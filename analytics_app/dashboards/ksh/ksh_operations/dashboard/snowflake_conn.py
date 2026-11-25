"""
Snowflake connection + cached query helper for KSH V3 Operations Dashboard.

RSA key-pair auth. Credentials from .env at repository root.
_find_root() walks up from this file so the module works from any working directory.
Targets HOSPITALS.REPORTING (gold layer).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import snowflake.connector
import streamlit as st
from dotenv import load_dotenv


_ROOT = Path(__file__).resolve().parents[4]  # dashboard/ksh_operations/ksh_v3_analysis/private_analysis/TKLK
load_dotenv(_ROOT / ".env")


def _key_path() -> str:
    raw = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "rsa_key.p8").strip()
    p = Path(raw)
    return str(p if p.is_absolute() else (_ROOT / p).resolve())


@st.cache_resource(show_spinner=False)
def _get_connection() -> "snowflake.connector.SnowflakeConnection":
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT", "").strip(),
        user=os.getenv("SNOWFLAKE_USER", "").strip(),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "").strip(),
        database=os.getenv("SNOWFLAKE_DATABASE", "HOSPITALS").strip(),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "REPORTING").strip(),
        private_key_file=_key_path(),
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


@st.cache_data(ttl=3600, show_spinner=False)
def run_query(sql: str) -> pd.DataFrame:
    """Execute SQL, return DataFrame with UPPERCASE columns. Cached 1 hour.
    On token expiry (390114), clears the connection cache and retries once."""
    try:
        return _execute(sql)
    except snowflake.connector.errors.ProgrammingError as e:
        if e.errno == 390114:
            _get_connection.clear()
            return _execute(sql)
        raise
