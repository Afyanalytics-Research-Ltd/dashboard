"""
Snowflake connection for KSH V3 Operations Dashboard.
Uses the shared SnowflakeClient from analytics_app/dashboards/snowflake_service/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# analytics_app/dashboards/ must be on sys.path for snowflake_service to be importable
_DASHBOARDS = Path(__file__).resolve().parents[3]
if str(_DASHBOARDS) not in sys.path:
    sys.path.insert(0, str(_DASHBOARDS))

from snowflake_service.snowflake_client import SnowflakeClient  # noqa: E402

_client = SnowflakeClient()


@st.cache_data(ttl=3600, show_spinner=False)
def run_query(sql: str) -> pd.DataFrame:
    """Execute SQL, return DataFrame with UPPERCASE columns. Cached 1 hour."""
    return _client.query(sql)
