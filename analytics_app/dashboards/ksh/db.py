import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from connect_to_snowflake import run_query as _run_query


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
