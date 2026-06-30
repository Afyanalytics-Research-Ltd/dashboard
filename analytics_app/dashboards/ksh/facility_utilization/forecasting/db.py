"""
Minimal Snowflake connector for the forecasting module.
No Streamlit dependency — safe to import from Django, trainer, or any CLI context.
Env vars are loaded by Django settings.py — do not load .env here.
"""

import os
import pandas as pd
import snowflake.connector
from cryptography.hazmat.primitives import serialization


def _load_private_key(path: str) -> bytes:
    with open(path, "rb") as f:
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
        cur = conn.cursor()
        cur.execute(sql)
        df = cur.fetch_pandas_all()
        df.columns = [c.lower() for c in df.columns]
        return df
    finally:
        conn.close()
