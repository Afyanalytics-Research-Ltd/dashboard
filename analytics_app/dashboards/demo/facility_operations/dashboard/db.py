import os
import pandas as pd
import snowflake.connector
import streamlit as st
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv
from pathlib import Path

def _find_root() -> Path:
    """Nearest ancestor directory containing a .env file (the repo root)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".env").exists():
            return parent
    return here.parents[5]


_ROOT = _find_root()
load_dotenv(_ROOT / ".env")


def _load_private_key(path: str) -> bytes:
    # Relative key paths in .env are relative to the .env's folder, not the cwd
    p = Path(path)
    if not p.is_absolute():
        p = _ROOT / p
    with open(p, "rb") as f:
        p_key = serialization.load_pem_private_key(f.read(), password=None)
    return p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def get_connection():
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        private_key=_load_private_key(os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE"),
    )


def run_query(sql: str) -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


@st.cache_data(ttl=3600, show_spinner=False)
def run_query_df(sql: str) -> pd.DataFrame:
    return run_query(sql)


def clear_cache():
    run_query_df.clear()
