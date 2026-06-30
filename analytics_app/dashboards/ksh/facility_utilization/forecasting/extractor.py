"""
KSH daily admission extractor.

Single responsibility: pull daily admission counts from Snowflake,
flag the confirmed EMR sync gap (Inv 32 + Inv 65), validate output, return DataFrame.

Does NOT write to PostgreSQL — that is the loader's job.
Does NOT aggregate to monthly — that is the feature builder's job.
Does NOT filter by ward — facility-level only in Phase 1.

Output schema:
    admission_date  : datetime64[ns]  — one row per day that had >= 1 admission
    admissions      : int64           — count of admissions that day
    is_gap_flagged  : bool            — True for Oct 4–16 2025 (EMR sync failure)
"""

import pandas as pd
from .db import run_query

# Oct 4–16 2025: confirmed EMR sync failure (Inv 32, Inv 65).
# ALL gap days are absent from source — no rows exist to flag here.
# is_gap_flagged will always be False from the extractor.
# The feature builder applies the flag when building the complete date spine.
_GAP_START = pd.Timestamp("2025-10-04")
_GAP_END   = pd.Timestamp("2025-10-16")

# Data starts Sep 2, 2024 (Inv 65). Sep 1 is Sunday — no admissions.
# Boundary set to end of Sep 2024 so any Sep start date passes.
_EXPECTED_START = pd.Timestamp("2024-09-30")

# Minimum rows expected — ~570 days of data, ~450+ days with admissions
_MIN_ROWS = 400

_SQL = """
    SELECT
        admission_date::DATE AS admission_date,
        COUNT(*)             AS admissions
    FROM HOSPITALS.STAGING.stg_inpatient_admissions
    WHERE source_schema = 'KISUMU_CLEAN'
      AND admission_date IS NOT NULL
    GROUP BY 1
    ORDER BY 1
"""


def pull() -> pd.DataFrame:
    """
    Pull daily KSH admissions from Snowflake.
    Returns a validated DataFrame. Raises ValueError if validation fails.
    """
    df = run_query(_SQL)

    df["admission_date"] = pd.to_datetime(df["admission_date"])
    df["admissions"]     = df["admissions"].astype("int64")

    df["is_gap_flagged"] = (
        (df["admission_date"] >= _GAP_START) &
        (df["admission_date"] <= _GAP_END)
    )

    df = df.sort_values("admission_date").reset_index(drop=True)

    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    """
    Hard checks on extractor output.
    Raises ValueError with a clear message if any check fails.
    Designed to catch schema changes, source truncation, or pipeline failures early.
    """
    required = {"admission_date", "admissions", "is_gap_flagged"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"extractor: missing columns: {missing}")

    if len(df) < _MIN_ROWS:
        raise ValueError(
            f"extractor: only {len(df)} rows returned — expected >= {_MIN_ROWS}. "
            "Source may be truncated or query failed silently."
        )

    if (df["admissions"] < 0).any():
        raise ValueError("extractor: negative admission counts found — data integrity issue.")

    if df["admission_date"].min() > _EXPECTED_START:
        raise ValueError(
            f"extractor: earliest date is {df['admission_date'].min().date()} — "
            f"expected data to start by {_EXPECTED_START.date()}. "
            "Source filter or table may have changed."
        )

    if df["admission_date"].duplicated().any():
        raise ValueError("extractor: duplicate dates found — GROUP BY may have failed.")

    if df["is_gap_flagged"].dtype != bool:
        raise ValueError("extractor: is_gap_flagged must be bool dtype.")
