# Import python packages
import streamlit as st
import os
from snowflake.snowpark.context import get_active_session
from utils import *
from snowflake.snowpark import Session
from dotenv import load_dotenv
load_dotenv()

connection_parameters = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT").strip(),
    "user": os.getenv("SNOWFLAKE_USER").strip(),
    "password": os.getenv("SNOWFLAKE_PASSWORD").strip(),
    "role": os.getenv("SNOWFLAKE_ROLE").strip(),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE").strip(),
    "database": os.getenv("SNOWFLAKE_DATABASE").strip(),
    "schema": os.getenv("SNOWFLAKE_SCHEMA").strip()
}

st.title(f"Revenue Analysis")

session = Session.builder.configs(connection_parameters).create()
df = load_events_table(session, "HOSPITALS.AFYA_API_AUTH_RAW.EVENTS_RAW")

finance_df = get_module_dataframe(df,"Finance")
financial_tables = get_module_tables(finance_df)

finance_invoices = get_source_table(df, "Finance", "finance_invoices")
finance_invoices = parse_payload_column(finance_invoices)

finance_waivers = get_source_table(df, "Finance", "finance_waivers")
finance_waivers = parse_payload_column(finance_waivers)

finance_invoices_df = normalize_first_payload(finance_invoices)
finance_waivers_df = normalize_first_payload(finance_waivers)

invoices_waivers = pd.merge(finance_invoices_df, finance_waivers_df, left_on="id", right_on="invoice_id")


st.dataframe(invoices_waivers.head())
