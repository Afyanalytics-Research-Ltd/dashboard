"""
Facility registry and session-state helpers.
Each facility is fully isolated — no "Both" view in operational pages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st


# ── Registry ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FacilityMeta:
    schema: str            # Snowflake source_schema discriminator
    label: str             # Display name
    short: str             # Short badge label
    is_live: bool          # True = active facility, False = historical
    date_range: str        # Human-readable data coverage
    alert_count_key: str   # Session-state key for badge on landing page
    # Snowflake table overrides (None = use global default)
    batch_purchases_table: Optional[str] = None
    # Fixed anchor date (YYYY-MM-DD) for all time-windowed queries. Set this
    # only to pin a facility to a hard historical cutoff. When left None, the
    # anchor is resolved dynamically to MAX(dispensed_at) in the facility's
    # data, so lookback windows (e.g. -90 days) always find real records even
    # when the data lags the calendar date.
    data_end_date: Optional[str] = None
    # Clinical go-live date (YYYY-MM-DD). Records before this date are
    # test/training data and should be excluded from patient-facing analytics.
    go_live_date: Optional[str] = None


FACILITIES: dict[str, FacilityMeta] = {
    "kisumu": FacilityMeta(
        schema="kisumu",
        label="Kisumu Specialist Hospital",
        short="KSH",
        is_live=True,
        date_range="Sep 2024 – Present",
        alert_count_key="ksh_alert_count",
        batch_purchases_table=None,  # no batch table; SOH-jump detection used
        data_end_date=None,          # dynamic — anchor to MAX(dispensed_at) in data
        go_live_date="2024-09-01",   # clinical go-live; pre-date records are test data
    ),

}


# ── Session-state helpers ─────────────────────────────────────────────────────

SESSION_KEY = "active_facility"


def get_active_facility() -> Optional[FacilityMeta]:
    """Return the currently selected FacilityMeta, or None if not yet chosen."""
    schema = st.session_state.get(SESSION_KEY)
    return FACILITIES.get(schema) if schema else None


def set_active_facility(schema: str) -> None:
    st.session_state[SESSION_KEY] = schema


def require_facility() -> FacilityMeta:
    """Return the active facility, defaulting to KSH if session state was cleared."""
    fac = get_active_facility()
    if fac is None:
        set_active_facility("kisumu")
        fac = get_active_facility()
    return fac


def sql_schema_filter(schema: str) -> str:
    """Return a safe single-value SQL IN clause: ('kisumu')"""
    return f"('{schema}')"


def sql_go_live_filter(fac: FacilityMeta, date_col: str = "dispensed_at") -> str:
    """
    Return a SQL AND-clause that excludes pre-go-live test records.
    Empty string if no go_live_date is set (safe to concatenate).

    Usage in queries:
        {sql_go_live_filter(fac)}   -- e.g. AND dispensed_at >= '2024-09-01'
    """
    if fac.go_live_date:
        return f"AND {date_col} >= '{fac.go_live_date}'"
    return ""


@st.cache_data(ttl=3600, show_spinner=False)
def _resolve_max_data_date(schema: str) -> Optional[str]:
    """
    Most recent dispensing date in a facility's data, as an ISO 'YYYY-MM-DD'
    string. Returns None if it can't be resolved (empty data or query error).

    Capped at CURRENT_DATE so a stray future-dated record can't push the
    anchor past today. Cached for 1 hour alongside the rest of the query layer.
    """
    try:
        from utils.snowflake_conn import run_query
        df = run_query(f"""
            SELECT TO_VARCHAR(
                       LEAST(MAX(dispensed_at)::DATE, CURRENT_DATE),
                       'YYYY-MM-DD') AS max_date
            FROM HOSPITALS.REPORTING.FACT_DISPENSING
            WHERE source_schema = '{schema}'
        """)
        if not df.empty and df.iloc[0, 0]:
            return str(df.iloc[0, 0])
    except Exception:
        pass
    return None


def sql_ref_date(fac: FacilityMeta) -> str:
    """
    Return the SQL date expression to use as the 'current date' anchor for
    all time-windowed queries.

    The data is not live-streamed — the newest dispensing record can lag the
    real calendar date by weeks or months. Anchoring lookback windows (e.g.
    -90 days) to CURRENT_DATE would then find no records and blank out
    Patient Risk, Stockout Watch and the briefing patient monitor. So we
    anchor to the most recent dispensing date in the facility's own data.

    - Explicit data_end_date set → use it (fixed historical facilities).
    - Otherwise                  → resolve MAX(dispensed_at) from the data.
    - On any failure             → fall back to CURRENT_DATE.

    Returns a plain quoted ISO literal ('YYYY-MM-DD') so parse_ref_date() and
    downstream `.strip("'")` handling keep working unchanged.
    """
    if fac.data_end_date:
        return f"'{fac.data_end_date}'"
    resolved = _resolve_max_data_date(fac.schema)
    return f"'{resolved}'" if resolved else "CURRENT_DATE"
