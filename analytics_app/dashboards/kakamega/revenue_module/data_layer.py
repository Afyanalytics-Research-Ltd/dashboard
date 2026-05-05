"""
data_layer.py
-------------
Streamlit-aware data layer. Wraps SnowflakeClient + queries.py and exposes
one function per analytic question. Every function is decorated with
`@st.cache_data` so the warehouse is hit at most once per (date range,
filter set) per session.

There is NO fallback to simulated data — all numbers come from Snowflake.
The dashboard surfaces a clear error if the warehouse is unreachable.

Public functions (each returns a pandas.DataFrame):
    daily_revenue           revenue_by_service_line   revenue_by_branch
    payment_mode_mix        payer_performance         patient_rfm
    top_items               hourly_heatmap            cohort_retention
    doctor_productivity     leakage                   inventory_margin
    claim_rejection         revenue_concentration     arpv_trend
    revenue_at_risk         gross_profit_weekly
"""



import streamlit as st
import pandas as pd

import queries
from snowflake_service.snowflake_client import SnowflakeClient


# ─── Connection ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_client(schema=None) -> SnowflakeClient:
    """Single Snowflake connection for the lifetime of the Streamlit session."""
    return SnowflakeClient(schema_=schema)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Snowflake returns column names UPPERCASE — lower them so the rest of
    the codebase can use the same names as the SQL aliases."""
    if df is None or len(df) == 0:
        return df
    df.columns = [c.lower() for c in df.columns]
    return df


def _run(query_name: str, *, start: str, end: str, clinic_filter: str = "") -> pd.DataFrame:
    sql = queries.render(query_name, start=start, end=end, clinic_filter=clinic_filter)
    df = _get_client(schema='KAKAMEGA_CLEAN').query(sql)
    return _normalise(df)


# ─── Cached fetch functions (one per analytic) ───────────────────────────────

@st.cache_data(ttl=900, show_spinner="Fetching daily revenue from Snowflake…")
def daily_revenue(start: str, end: str, clinic_filter: str = "") -> pd.DataFrame:
    df = _run("daily_revenue", start=start, end=end, clinic_filter=clinic_filter)
    if "revenue_date" in df.columns:
        df["revenue_date"] = pd.to_datetime(df["revenue_date"])
    return df


@st.cache_data(ttl=900, show_spinner="Fetching service-line revenue…")
def revenue_by_service_line(start: str, end: str) -> pd.DataFrame:
    df = _run("revenue_by_service_line", start=start, end=end)
    if "revenue_month" in df.columns:
        df["revenue_month"] = pd.to_datetime(df["revenue_month"])
    return df


@st.cache_data(ttl=900, show_spinner="Fetching branch revenue…")
def revenue_by_branch(start: str, end: str) -> pd.DataFrame:
    df = _run("revenue_by_branch", start=start, end=end)
    if "revenue_month" in df.columns:
        df["revenue_month"] = pd.to_datetime(df["revenue_month"])
    return df


@st.cache_data(ttl=900, show_spinner="Fetching payment-mode mix…")
def payment_mode_mix(start: str, end: str) -> pd.DataFrame:
    df = _run("payment_mode_mix", start=start, end=end)
    if "revenue_month" in df.columns:
        df["revenue_month"] = pd.to_datetime(df["revenue_month"])
    return df


@st.cache_data(ttl=900, show_spinner="Fetching payer performance…")
def payer_performance(start: str, end: str) -> pd.DataFrame:
    return _run("payer_performance", start=start, end=end)


@st.cache_data(ttl=900, show_spinner="Fetching patient RFM table…")
def patient_rfm(start: str, end: str) -> pd.DataFrame:
    df = _run("patient_rfm", start=start, end=end)
    for c in ("last_receipt", "first_receipt"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


@st.cache_data(ttl=900, show_spinner="Fetching top items…")
def top_items(start: str, end: str) -> pd.DataFrame:
    return _run("top_items", start=start, end=end)


@st.cache_data(ttl=900, show_spinner="Fetching hourly heatmap…")
def hourly_heatmap(start: str, end: str) -> pd.DataFrame:
    return _run("hourly_heatmap", start=start, end=end)


@st.cache_data(ttl=900, show_spinner="Fetching cohort retention…")
def cohort_retention(start: str, end: str) -> pd.DataFrame:
    df = _run("cohort_retention", start=start, end=end)
    for c in ("cohort_month", "active_month"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


@st.cache_data(ttl=900, show_spinner="Fetching doctor productivity…")
def doctor_productivity(start: str, end: str) -> pd.DataFrame:
    return _run("doctor_productivity", start=start, end=end)


@st.cache_data(ttl=900, show_spinner="Fetching leakage…")
def leakage(start: str, end: str) -> pd.DataFrame:
    df = _run("leakage", start=start, end=end)
    if "revenue_month" in df.columns:
        df["revenue_month"] = pd.to_datetime(df["revenue_month"])
    return df


@st.cache_data(ttl=900, show_spinner="Fetching inventory margin…")
def inventory_margin(start: str, end: str) -> pd.DataFrame:
    return _run("inventory_margin", start=start, end=end)


@st.cache_data(ttl=900, show_spinner="Fetching claim rejection…")
def claim_rejection(start: str, end: str) -> pd.DataFrame:
    df = _run("claim_rejection", start=start, end=end)
    if "revenue_month" in df.columns:
        df["revenue_month"] = pd.to_datetime(df["revenue_month"])
    return df


@st.cache_data(ttl=900, show_spinner="Fetching revenue concentration…")
def revenue_concentration(start: str, end: str) -> pd.DataFrame:
    return _run("revenue_concentration", start=start, end=end)


@st.cache_data(ttl=900, show_spinner="Fetching ARPV trend…")
def arpv_trend(start: str, end: str) -> pd.DataFrame:
    df = _run("arpv_trend", start=start, end=end)
    if "revenue_date" in df.columns:
        df["revenue_date"] = pd.to_datetime(df["revenue_date"])
    return df


@st.cache_data(ttl=900, show_spinner="Fetching revenue at risk…")
def revenue_at_risk(start: str, end: str) -> pd.DataFrame:
    return _run("revenue_at_risk", start=start, end=end)


@st.cache_data(ttl=900, show_spinner="Fetching gross profit…")
def gross_profit_weekly(start: str, end: str) -> pd.DataFrame:
    df = _run("gross_profit_weekly", start=start, end=end)
    if "week" in df.columns:
        df["week"] = pd.to_datetime(df["week"])
    return df


# ─── Convenience: fetch a clinic dimension once for the sidebar filter ──────

@st.cache_data(ttl=3600, show_spinner=False)
def list_clinics() -> pd.DataFrame:
    sql = """
        SELECT ID AS clinic_id, NAME AS clinic_name, TOWN AS town
        FROM SETTINGS_CLINICS
        WHERE STATUS = 'active' AND DELETED_AT IS NULL
        ORDER BY clinic_name
    """
    return _normalise(_get_client().query(sql))