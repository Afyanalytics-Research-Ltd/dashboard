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
    # For historical facilities: last date with data (YYYY-MM-DD).
    # All time-windowed queries anchor to this date instead of CURRENT_DATE
    # so that lookback windows (e.g. -90 days) actually find records.
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
        data_end_date=None,          # live — use CURRENT_DATE
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


def sql_ref_date(fac: FacilityMeta) -> str:
    """
    Return the SQL date expression to use as the 'current date' anchor for
    all time-windowed queries.

    - Live facilities  → CURRENT_DATE  (real-time)
    - Historical facilities → literal date string anchored to the last day
      of data, so that lookback windows (e.g. -90 days) find real records
      instead of returning empty.
    """
    if fac.is_live or fac.data_end_date is None:
        return "CURRENT_DATE"
    return f"'{fac.data_end_date}'::DATE"
