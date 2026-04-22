"""
PharmaPlus Kenya — Seasonal Signal Engine
==========================================
Applies seasonal flags and discount boost multipliers to df_expiry
based on a configurable promotional calendar.

Usage:
    from src.seasonal import apply_seasonal_signals
    df_enriched = apply_seasonal_signals(df_expiry, today=date.today())

Output columns added / updated:
    seasonal_flag   : str | None  — name of the active season (or None)
    seasonal_boost  : float       — velocity / discount multiplier [1.0, 2.0]
"""

from datetime import date
from typing import Optional

import polars as pl


# ── Seasonal calendar ──────────────────────────────────────────────────────────
# Each entry: (month_start, day_start, month_end, day_end, label, boost)
# Boost is applied as a multiplier to daily_velocity for demand estimation.
# Overlapping windows: later entries take priority (last-wins in the scan).

SEASONAL_CALENDAR = [
    # Valentines
    (2,  1,  2, 14, "Valentines",        1.25),
    # Easter / Pasaka (approximate — April, varies yearly)
    (4,  1,  4, 25, "Easter",            1.20),
    # Mother's Day — May
    (5,  1,  5, 15, "Mothers Day",       1.20),
    # Mid-year clearance
    (6, 15,  7, 15, "Mid-Year Sale",     1.30),
    # Back to school — January & August
    (1,  1,  1, 20, "Back to School",    1.15),
    (8,  1,  8, 20, "Back to School",    1.15),
    # Black Friday / Cyber Monday
    (11, 20, 11, 30, "Black Friday",     1.40),
    # December festive
    (12,  1, 12, 31, "December Festive", 1.50),
    # New Year
    (1,   1,  1,  7, "New Year",         1.20),
]


def _active_season(
    ref_date: date,
    calendar: list = SEASONAL_CALENDAR,
) -> tuple:
    """
    Returns (label, boost) for the first matching season on ref_date,
    or (None, 1.0) if no season is active.

    Later entries in calendar take priority (last-wins scan via reversed).
    """
    result_label = None
    result_boost = 1.0

    for (ms, ds, me, de, label, boost) in calendar:
        season_start = date(ref_date.year, ms, ds)
        # Handle year-wrap (e.g. Dec 25 → Jan 5)
        if me < ms or (me == ms and de < ds):
            season_end = date(ref_date.year + 1, me, de)
        else:
            season_end = date(ref_date.year, me, de)

        if season_start <= ref_date <= season_end:
            result_label = label
            result_boost = boost  # last-wins

    return result_label, result_boost


def apply_seasonal_signals(
    df: pl.DataFrame,
    today: Optional[date] = None,
    calendar: list = SEASONAL_CALENDAR,
) -> pl.DataFrame:
    """
    Enrich df_expiry with seasonal_flag and seasonal_boost.

    Args:
        df      : expiry stock DataFrame (must have daily_velocity)
        today   : reference date (defaults to date.today())
        calendar: list of seasonal windows (defaults to SEASONAL_CALENDAR)

    Returns:
        df with seasonal_flag (str | null) and seasonal_boost (float) columns.
        seasonal_boost is 1.0 outside any promotional window.
    """
    if today is None:
        today = date.today()

    label, boost = _active_season(today, calendar)

    df = df.with_columns([
        pl.lit(label).alias("seasonal_flag"),
        pl.lit(boost).alias("seasonal_boost"),
    ])

    return df


def get_upcoming_seasons(
    today: Optional[date] = None,
    lookahead_days: int = 90,
    calendar: list = SEASONAL_CALENDAR,
) -> list:
    """
    Returns a list of upcoming seasons within lookahead_days.
    Useful for dashboard awareness widgets.

    Returns:
        list of dicts with keys: label, start, end, boost, days_away
    """
    if today is None:
        today = date.today()

    from datetime import timedelta
    horizon = today + timedelta(days=lookahead_days)
    upcoming = []

    for (ms, ds, me, de, label, boost) in calendar:
        season_start = date(today.year, ms, ds)
        if season_start < today:
            # Check if it recurs next year within window
            season_start = date(today.year + 1, ms, ds)

        if me < ms or (me == ms and de < ds):
            season_end = date(season_start.year + 1, me, de)
        else:
            season_end = date(season_start.year, me, de)

        if today <= season_start <= horizon:
            upcoming.append({
                "label":     label,
                "start":     season_start.isoformat(),
                "end":       season_end.isoformat(),
                "boost":     boost,
                "days_away": (season_start - today).days,
            })

    return sorted(upcoming, key=lambda x: x["days_away"])
