import os
import pandas as pd
import streamlit as st

from snowflake_service.snowflake_client import SnowflakeClient

@st.cache_resource
def _get_client() -> SnowflakeClient:
    return SnowflakeClient()
 
 
@st.cache_data(ttl=3600, show_spinner=False)
def run_query(query: str) -> pd.DataFrame:
    return _get_client().query(query)