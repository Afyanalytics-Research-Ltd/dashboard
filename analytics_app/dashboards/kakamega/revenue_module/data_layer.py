"""
data_layer.py
-------------
Streamlit-aware data layer. Wraps SnowflakeClient + queries.py and exposes
one function per analytic question. Every function is decorated with
`@st.cache_data` so the warehouse is hit at most once per (date range,
filter set) per session.

There is NO fallback to simulated data — all numbers come from Snowflake.
The dashboard surfaces a clear error if the warehouse is unreachable.

Logging
~~~~~~~
Each query is logged with a one-line header (▶) on dispatch and a one-line
footer (✓ / ✗) on completion. Footer carries timing + row count, so a typical
session looks like:

    ▶ daily_revenue            start=2025-05-05 end=2026-05-06 schema=KISUMU_CLEAN
    ✓ daily_revenue            12,840 rows · 1.42s
    ▶ revenue_by_branch        start=2025-05-05 end=2026-05-06 schema=KISUMU_CLEAN
    ✗ revenue_by_branch        Snowflake error after 0.31s
        SQL compilation error: Function DATE_TRUNC does not support …
        — first 12 lines of failing SQL —

Public functions (each returns a pandas.DataFrame):
    daily_revenue           revenue_by_service_line   revenue_by_branch
    payment_mode_mix        payer_performance         patient_rfm
    top_items               hourly_heatmap            cohort_retention
    doctor_productivity     leakage                   inventory_margin
    claim_rejection         revenue_concentration     arpv_trend
    revenue_at_risk         gross_profit_weekly       list_clinics
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional

import pandas as pd
import streamlit as st

import kakamega.revenue_module.queries as queries
from snowflake_service.snowflake_client import SnowflakeClient


# ─── Logging setup ──────────────────────────────────────────────────────────

# ANSI colors — auto-disabled if NO_COLOR is set or stdout isn't a TTY.
def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty() or os.environ.get("FORCE_COLOR") == "1"


_USE_COLOR = _supports_color()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


# Color helpers
def _dim(s):     return _c(s, "2")
def _bold(s):    return _c(s, "1")
def _cyan(s):    return _c(s, "36")
def _green(s):   return _c(s, "32")
def _red(s):     return _c(s, "31")
def _yellow(s):  return _c(s, "33")
def _magenta(s): return _c(s, "35")
def _blue(s):    return _c(s, "34")


class _DataLayerFormatter(logging.Formatter):
    """Compact formatter: HH:MM:SS  level  message."""
    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        level_color = {
            "INFO":     _cyan,
            "WARNING":  _yellow,
            "ERROR":    _red,
            "DEBUG":    _dim,
            "CRITICAL": _red,
        }.get(record.levelname, lambda s: s)
        prefix = f"{_dim(ts)}  {level_color(record.levelname.ljust(7))}"
        return f"{prefix}  {record.getMessage()}"


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("revenue.data_layer")
    if logger.handlers:                     # Streamlit reruns — don't double-attach
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_DataLayerFormatter())
    logger.addHandler(handler)
    return logger


log = _build_logger()


# ─── Connection ──────────────────────────────────────────────────────────────
def _get_client(schema: Optional[str] = None) -> SnowflakeClient:
    """Single Snowflake connection per (session, schema)."""
    log.info(
        "Opening Snowflake connection · schema=%s",
        _bold(schema or os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")),
    )
    try:
        return SnowflakeClient(schema_=schema)
    except Exception as exc:
        log.error("Snowflake connection failed: %s", _red(str(exc)))
        raise


# ─── DataFrame helpers ──────────────────────────────────────────────────────
from decimal import Decimal
def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Snowflake returns column names UPPERCASE — lower them so the rest of
    the codebase can use the same names as the SQL aliases."""
    if df is None or len(df) == 0:
        return df
    
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna()
            if len(sample) and isinstance(sample.iloc[0], Decimal):
                df[col] = pd.to_numeric(df[col], errors="coerce")

    df.columns = [c.lower() for c in df.columns]
    return df


def _format_sql_excerpt(sql: str, lines: int = 12) -> str:
    """First N lines of the SQL, indented, for error reporting."""
    head = sql.strip().splitlines()[:lines]
    return "\n".join("    " + line for line in head)


# ─── Query runner with structured logging ───────────────────────────────────

def _run(
    query_name: str,
    *,
    start: str,
    end: str,
    clinic_filter: str = "",
    schema: str = "KAKAMEGA_CLEAN",
) -> pd.DataFrame:
    """Render → execute → normalise. All logging happens here."""
    sql = queries.render(query_name, start=start, end=end, clinic_filter=clinic_filter)

    label = _cyan(_bold(query_name.ljust(26)))
    log.info(
        "%s  %s  start=%s  end=%s  schema=%s",
        _green("▶"), label, _bold(start), _bold(end), _magenta(schema),
    )

    t0 = time.perf_counter()
    try:
        df = _get_client(schema=schema).query(sql)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        log.error(
            "%s  %s  failed after %s",
            _red("✗"), label, _yellow(f"{elapsed:.2f}s"),
        )
        log.error("        %s", _red(str(exc).splitlines()[0]))
        log.error("%s\n%s", _dim("        — failing SQL excerpt —"), _dim(_format_sql_excerpt(sql)))
        raise

    df = _normalise(df)
    elapsed = time.perf_counter() - t0
    n = len(df) if df is not None else 0
    log.info(
        "%s  %s  %s rows · %s",
        _green("✓"), label, _bold(f"{n:,}"), _yellow(f"{elapsed:.2f}s"),
    )
    return df


# ─── Cached fetch functions ─────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner="Fetching daily revenue from Snowflake…")
def daily_revenue(start: str, end: str, clinic_filter: str = "") -> pd.DataFrame:
    df = _run("daily_revenue", start=start, end=end, clinic_filter=clinic_filter)
    if "revenue_date" in df.columns:
        df["revenue_date"] = pd.to_datetime(df["revenue_date"], errors="coerce")
    return df


@st.cache_data(ttl=900, show_spinner="Fetching service-line revenue…")
def revenue_by_service_line(start: str, end: str) -> pd.DataFrame:
    df = _run("revenue_by_service_line", start=start, end=end)
    if "revenue_month" in df.columns:
        df["revenue_month"] = pd.to_datetime(df["revenue_month"], errors="coerce")
    return df



@st.cache_data(ttl=900, show_spinner="Fetching payment-mode mix…")
def payment_mode_mix(start: str, end: str) -> pd.DataFrame:
    df = _run("payment_mode_mix", start=start, end=end)
    if "revenue_month" in df.columns:
        df["revenue_month"] = pd.to_datetime(df["revenue_month"], errors="coerce")
    return df


@st.cache_data(ttl=900, show_spinner="Fetching payer performance…")
def payer_performance(start: str, end: str) -> pd.DataFrame:
    return _run("payer_performance", start=start, end=end)


@st.cache_data(ttl=900, show_spinner="Fetching patient RFM table…")
def patient_rfm(start: str, end: str) -> pd.DataFrame:
    df = _run("patient_rfm", start=start, end=end)
    for c in ("last_receipt", "first_receipt"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
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
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


@st.cache_data(ttl=900, show_spinner="Fetching doctor productivity…")
def doctor_productivity(start: str, end: str) -> pd.DataFrame:
    return _run("doctor_productivity", start=start, end=end)


@st.cache_data(ttl=900, show_spinner="Fetching leakage…")
def leakage(start: str, end: str) -> pd.DataFrame:
    df = _run("leakage", start=start, end=end)
    if "revenue_month" in df.columns:
        df["revenue_month"] = pd.to_datetime(df["revenue_month"], errors="coerce")
    return df




@st.cache_data(ttl=900, show_spinner="Fetching claim rejection…")
def claim_rejection(start: str, end: str) -> pd.DataFrame:
    df = _run("claim_rejection", start=start, end=end)
    if "revenue_month" in df.columns:
        df["revenue_month"] = pd.to_datetime(df["revenue_month"], errors="coerce")
    return df


@st.cache_data(ttl=900, show_spinner="Fetching revenue concentration…")
def revenue_concentration(start: str, end: str) -> pd.DataFrame:
    return _run("revenue_concentration", start=start, end=end)


@st.cache_data(ttl=900, show_spinner="Fetching ARPV trend…")
def arpv_trend(start: str, end: str) -> pd.DataFrame:
    df = _run("arpv_trend", start=start, end=end)
    if "revenue_date" in df.columns:
        df["revenue_date"] = pd.to_datetime(df["revenue_date"], errors="coerce")
    return df


@st.cache_data(ttl=900, show_spinner="Fetching revenue at risk…")
def revenue_at_risk(start: str, end: str) -> pd.DataFrame:
    return _run("revenue_at_risk", start=start, end=end)




# ─── Convenience: clinic dimension for the sidebar filter ──────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def list_clinics(schema: str = "KISUMU_CLEAN") -> pd.DataFrame:
    sql = """
        SELECT ID AS clinic_id, NAME AS clinic_name, TOWN AS town
        FROM SETTINGS_CLINICS
        WHERE STATUS = 'active' AND DELETED_AT IS NULL
        ORDER BY clinic_name
    """
    label = _cyan(_bold("list_clinics".ljust(26)))
    log.info("%s  %s  schema=%s", _green("▶"), label, _magenta(schema))
    t0 = time.perf_counter()
    try:
        df = _normalise(_get_client(schema=schema).query(sql))
    except Exception as exc:
        log.error("%s  %s  %s", _red("✗"), label, _red(str(exc).splitlines()[0]))
        raise
    elapsed = time.perf_counter() - t0
    log.info(
        "%s  %s  %s rows · %s",
        _green("✓"), label, _bold(f"{len(df):,}"), _yellow(f"{elapsed:.2f}s"),
    )
    if df.empty:
        log.warning("⚠️ No active clinics found in schema=%s", schema)
        return pd.DataFrame(columns=["clinic_id", "clinic_name", "town"])
    return df