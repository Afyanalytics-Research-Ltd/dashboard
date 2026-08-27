import os
import streamlit as st
import snowflake.connector
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from cryptography.hazmat.primitives import serialization

load_dotenv(find_dotenv())

_SOURCE = "MOTHERFRANSICA_MEDICAL_CAMP"


def _load_private_key(path: str) -> bytes:
    with open(path, "rb") as f:
        p_key = serialization.load_pem_private_key(f.read(), password=None)
    return p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _connect() -> snowflake.connector.SnowflakeConnection:
    private_key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "").strip()
    kwargs = dict(
        user=os.getenv("SNOWFLAKE_USER", "").strip(),
        account=os.getenv("SNOWFLAKE_ACCOUNT", "").strip(),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "").strip(),
        database=os.getenv("SNOWFLAKE_DATABASE", "HOSPITALS").strip(),
        schema="PUBLIC",
    )
    role = os.getenv("SNOWFLAKE_ROLE", "").strip()
    if role:
        kwargs["role"] = role
    if private_key_path:
        kwargs["private_key"] = _load_private_key(private_key_path)
    else:
        kwargs["password"] = os.getenv("SNOWFLAKE_PASSWORD", "").strip()
    return snowflake.connector.connect(**kwargs)


def _query(sql: str) -> pd.DataFrame:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return cur.fetch_pandas_all()
    finally:
        conn.close()


@st.cache_data(ttl=3600)
def get_encounters() -> pd.DataFrame:
    return _query(f"""
        SELECT * FROM HOSPITALS.REPORTING.rpt_mf_encounters
        WHERE source_schema = '{_SOURCE}'
    """)


@st.cache_data(ttl=3600)
def get_demographics() -> pd.DataFrame:
    return _query(f"""
        SELECT * FROM HOSPITALS.REPORTING.rpt_mf_demographics
        WHERE source_schema = '{_SOURCE}'
    """)


@st.cache_data(ttl=3600)
def get_diagnoses() -> pd.DataFrame:
    return _query(f"""
        SELECT * FROM HOSPITALS.REPORTING.rpt_mf_diagnoses
        WHERE source_schema = '{_SOURCE}'
        ORDER BY encounter_count DESC
    """)


@st.cache_data(ttl=3600)
def get_investigations() -> pd.DataFrame:
    return _query(f"""
        SELECT * FROM HOSPITALS.REPORTING.rpt_mf_investigations
        WHERE source_schema = '{_SOURCE}'
        ORDER BY encounter_count DESC
    """)


@st.cache_data(ttl=3600)
def get_drug_classes() -> pd.DataFrame:
    return _query(f"""
        SELECT * FROM HOSPITALS.REPORTING.rpt_mf_drug_classes
        WHERE source_schema = '{_SOURCE}'
        ORDER BY encounter_count DESC
    """)


@st.cache_data(ttl=3600)
def get_referrals() -> pd.DataFrame:
    return _query(f"""
        SELECT * FROM HOSPITALS.REPORTING.rpt_mf_referrals
        WHERE source_schema = '{_SOURCE}'
        ORDER BY encounter_count DESC
    """)


@st.cache_data(ttl=3600)
def get_procedures() -> pd.DataFrame:
    return _query(f"""
        SELECT * FROM HOSPITALS.REPORTING.rpt_mf_procedures
        WHERE source_schema = '{_SOURCE}'
        ORDER BY encounter_count DESC
    """)


@st.cache_data(ttl=3600)
def get_patient_spine() -> pd.DataFrame:
    return _query(f"""
        SELECT * FROM HOSPITALS.REPORTING.rpt_mf_patient_spine
        WHERE source_schema = '{_SOURCE}'
    """)


@st.cache_data(ttl=3600)
def get_patient_diagnoses() -> pd.DataFrame:
    return _query(f"""
        SELECT * FROM HOSPITALS.REPORTING.rpt_mf_patient_diagnoses
        WHERE source_schema = '{_SOURCE}'
        ORDER BY encounter_count DESC
    """)


@st.cache_data(ttl=3600)
def get_patient_medications() -> pd.DataFrame:
    return _query(f"""
        SELECT * FROM HOSPITALS.REPORTING.rpt_mf_patient_medications
        WHERE source_schema = '{_SOURCE}'
        ORDER BY encounter_count DESC
    """)


@st.cache_data(ttl=3600)
def get_patient_investigations() -> pd.DataFrame:
    return _query(f"""
        SELECT * FROM HOSPITALS.REPORTING.rpt_mf_patient_investigations
        WHERE source_schema = '{_SOURCE}'
        ORDER BY encounter_count DESC
    """)


@st.cache_data(ttl=3600)
def get_patient_referrals() -> pd.DataFrame:
    return _query(f"""
        SELECT * FROM HOSPITALS.REPORTING.rpt_mf_patient_referrals
        WHERE source_schema = '{_SOURCE}'
        ORDER BY encounter_count DESC
    """)


@st.cache_data(ttl=3600)
def get_age_diagnosis() -> pd.DataFrame:
    return _query(f"""
        SELECT * FROM HOSPITALS.REPORTING.rpt_mf_age_diagnosis
        WHERE source_schema = '{_SOURCE}'
        ORDER BY age_band, encounter_count DESC
    """)


@st.cache_data(ttl=3600)
def get_condition_profile() -> pd.DataFrame:
    return _query(f"""
        SELECT * FROM HOSPITALS.REPORTING.rpt_mf_condition_profile
        WHERE source_schema = '{_SOURCE}'
        ORDER BY patients_with_dx DESC
    """)


@st.cache_data(ttl=3600)
def get_vitals_signals() -> pd.DataFrame:
    return _query(f"""
        SELECT * FROM HOSPITALS.REPORTING.rpt_mf_vitals_signals
        WHERE source_schema = '{_SOURCE}'
    """)
