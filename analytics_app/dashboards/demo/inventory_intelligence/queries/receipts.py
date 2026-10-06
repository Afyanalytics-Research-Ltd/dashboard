"""
Receipt event queries — dual source.

KSH:   No batch purchase table exists. Receipt events detected from SOH jumps
       in fact_dispensing by the LeadTimeEngine (Python-side, not SQL).
       This module provides the raw dispensing data needed for that detection.

"""

from __future__ import annotations

import pandas as pd

from utils.snowflake_conn import run_query


def get_kisumu_dispensing_for_lead_time() -> pd.DataFrame:
    """
    Return KSH dispensing history with SOH columns needed by LeadTimeEngine.fit_kisumu().
    Loads the full history so the engine can detect all receipt events.
    """
    return run_query("""
        SELECT
            product_id,
            dispensed_at,
            soh_before,
            soh_after_raw,
            soh_after_display
        FROM HOSPITALS.REPORTING.FACT_DISPENSING
        WHERE source_schema = 'kisumu'
        ORDER BY product_id, dispensed_at
    """)


