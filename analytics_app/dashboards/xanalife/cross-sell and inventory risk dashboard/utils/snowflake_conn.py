import streamlit as st
import pandas as pd
import snowflake.connector


@st.cache_resource
def get_connection():
    cfg = st.secrets["snowflake"]
    return snowflake.connector.connect(
        account=cfg["account"],
        user=cfg["user"],
        warehouse=cfg["warehouse"],
        database=cfg["database"],
        schema=cfg["schema"],
        authenticator="programmatic_access_token",
        token=cfg["token"],
    )


@st.cache_data(ttl=3600, show_spinner=False)
def run_query(query: str) -> pd.DataFrame:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query)
    cols = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)
