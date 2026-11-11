"""
Snowflake connection + cached query helper.

Connection uses key-pair (RSA) auth. Credentials come from a `.env` file at the
repository root; the private key path may be absolute or relative to that root.
The `.env` is located by walking up from this file, so the same module works
whether it runs as a standalone repo or inside the wider analytics platform.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import snowflake.connector
import streamlit as st
from dotenv import load_dotenv


def _find_root() -> Path:
    """Nearest ancestor directory containing a .env file (fallback: repo root)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".env").exists():
            return parent
    # fallback: two levels above the package (…/<repo>/ksh_revenue/utils → <repo>)
    return here.parents[2]


_ROOT = _find_root()
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


@st.cache_data(ttl=3600, show_spinner=False)
def run_query(sql: str) -> pd.DataFrame:
    """Execute SQL and return a DataFrame (UPPERCASE columns). Cached 1 hour."""
    conn = _get_connection()
    cur = conn.cursor()
    try:
        cur.execute(sql)
        cols = [d[0].upper() for d in cur.description]
        rows = cur.fetchall()
    finally:
        cur.close()
    return pd.DataFrame(rows, columns=cols)
