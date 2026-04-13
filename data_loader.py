import os
import pandas as pd
import numpy as np
import streamlit as st
from snowflake.snowpark import Session
from utils import (
    load_events_table,
    parse_payload_column,
    get_module_dataframe,
    get_module_tables,
    get_source_table,
    normalize_all_payloads,
)

ALL_MODULES = [
    "Users", "Theatre", "Settings", "Reception", "Reports",
    "Evaluation", "Finance", "Inpatient", "Core", "Inventory",
]

@st.cache_resource
def get_session():
    params = {
        "account":   os.getenv("SNOWFLAKE_ACCOUNT", "").strip(),
        "user":      os.getenv("SNOWFLAKE_USER", "").strip(),
        "password":  os.getenv("SNOWFLAKE_PASSWORD", "").strip(),
        "role":      os.getenv("SNOWFLAKE_ROLE", "").strip(),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "").strip(),
        "database":  os.getenv("SNOWFLAKE_DATABASE", "").strip(),
        "schema":    os.getenv("SNOWFLAKE_SCHEMA", "").strip(),
    }
    return Session.builder.configs(params).create()


@st.cache_data(show_spinner=False)
def load_raw_events():
    session = get_session()
    return load_events_table(session, "HOSPITALS.AFYA_API_AUTH_RAW.EVENTS_RAW")


@st.cache_data(show_spinner=False)
def load_module_tables_map():
    raw = load_raw_events()
    result = {}
    for module in ALL_MODULES:
        try:
            mod_df = get_module_dataframe(raw, module)
            tables = list(get_module_tables(mod_df))
            result[module] = tables
        except Exception:
            result[module] = []
    return result


@st.cache_data(show_spinner=False)
def load_table(module: str, table: str) -> pd.DataFrame:
    raw = load_raw_events()
    try:
        src = get_source_table(raw, module, table)
        src = parse_payload_column(src)
        df = normalize_all_payloads(src)
        df = flatten_columns(df)
        df = auto_cast(df)
        return df
    except Exception:
        return pd.DataFrame()


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Stringify any nested dict/list columns so PyArrow can handle them."""
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(10)
            if sample.apply(lambda x: isinstance(x, (dict, list))).any():
                df[col] = df[col].apply(
                    lambda x: str(x) if isinstance(x, (dict, list)) else x
                )
    return df


def auto_cast(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt smart type inference on string columns."""
    date_hints = ["at", "date", "time", "created", "updated", "paid", "due", "start", "end"]
    num_hints  = ["amount", "balance", "paid", "total", "cost", "price", "quantity",
                  "count", "score", "rate", "id", "duration", "age", "weight"]
    for col in df.columns:
        if df[col].dtype != object:
            continue
        col_lower = col.lower()
        if any(h in col_lower for h in date_hints):
            converted = pd.to_datetime(df[col], errors="coerce", utc=True)
            if converted.notna().sum() > len(df) * 0.3:
                df[col] = converted
                continue
        if any(h in col_lower for h in num_hints):
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > len(df) * 0.3:
                df[col] = converted
    return df


def num_cols(df: pd.DataFrame) -> list:
    return df.select_dtypes(include="number").columns.tolist()


def date_cols(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]", "datetimetz"]).columns.tolist()


def cat_cols(df: pd.DataFrame, max_unique: int = 200) -> list:
    return [c for c in df.select_dtypes(include="object").columns
            if df[c].nunique() <= max_unique]


def best_amount_col(df: pd.DataFrame):
    for c in ["amount", "total_amount", "balance", "cost", "price", "paid"]:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    nums = num_cols(df)
    return nums[0] if nums else None


def best_date_col(df: pd.DataFrame):
    for c in ["created_at", "date", "start_date", "updated_at", "paid_at"]:
        if c in df.columns:
            return c
    dc = date_cols(df)
    return dc[0] if dc else None


def best_id_col(df: pd.DataFrame):
    for c in ["id", "patient_id", "user_id", "company_id"]:
        if c in df.columns:
            return c
    return df.columns[0] if len(df.columns) else None


def safe_show(df: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    """Ensure dataframe is safe to pass to st.dataframe."""
    out = df.head(n).copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str)
    return out
