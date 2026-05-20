import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from snowflake_service.snowflake_client import SnowflakeClient
_run_query = SnowflakeClient().query


@st.cache_data(ttl=3600, show_spinner=False)
def run_query_df(sql: str):
    """Execute SQL against HOSPITALS.REPORTING, return a DataFrame.

    Cached for 1 hour per unique SQL string.
    Column names are uppercase (Snowflake default) — use df["COLUMN_NAME"].
    """
    return _run_query(sql)


def clear_cache():
    """Force-refresh all cached queries. Call when filters change."""
    run_query_df.clear()
