"""
KSH admission feature builder.

Single responsibility: take sparse extractor output, build a complete daily date
spine, apply gap flags, fill legitimate zero-admission days, and engineer features
for model training and evaluation.

Does NOT write to PostgreSQL — that is the loader's job.
Does NOT train models — that is baseline_model.py / prophet_model.py.

Output functions
----------------
build(df)                  → full daily feature matrix (DataFrame)
prepare_prophet_inputs(df) → (ds, y) DataFrame ready for Prophet, gap rows excluded
make_prophet_holidays()    → Prophet-compatible Kenya holidays DataFrame

Zero-admission vs missing-data disambiguation
---------------------------------------------
Extractor SQL: SELECT admission_date::DATE, COUNT(*) ... GROUP BY 1
A date absent from extractor output means COUNT(*) = 0, i.e. no records at all
for that date in Snowflake. This is a genuine zero-admission day, not a data gap.
The ONLY confirmed data gap is Oct 4–16 2025 (Inv 32, Inv 65 — EMR sync failure).

After merge:
  non-gap absent dates → admissions = 0.0    (real zeros, used in training)
  gap dates            → admissions = NaN     (data unavailable, excluded from training)

This means prepare_prophet_inputs includes y=0 days — Prophet models them as
real zero-admission days, which is correct.

NaN propagation from gap
------------------------
The 13-day gap (Oct 4–16) propagates NaN into lag/rolling features for ~40 days:
  lag_7            → NaN for Oct 17–23  (7 days after gap)
  lag_14           → NaN for Oct 17–30  (14 days after gap)
  lag_28           → NaN for Nov 1–13   (28 days after gap)
  rolling_7d_mean  → NaN for Oct 5–23   (within + 7 days after gap)
  rolling_14d_mean → NaN for Oct 5–30
Trainer must exclude all rows where training target is NaN (gap rows) OR where
lag features are NaN (first 28 rows + gap-adjacent rows).

Feature matrix columns
----------------------
admission_date   : datetime64[ns]  — one row per calendar day
admissions       : float64         — 0.0 for zero days, NaN for gap days only
is_gap_flagged   : bool            — True for Oct 4–16 2025 only
day_of_week      : int8            — 0=Mon, 6=Sun
week_of_year     : int8            — ISO week number
month            : int8            — 1–12
is_kenya_holiday : bool            — Kenya public holidays (holidays library)
lag_7            : float64         — admissions 7 days prior
lag_14           : float64         — admissions 14 days prior
lag_28           : float64         — admissions 28 days prior
rolling_7d_mean  : float64         — mean of admissions over prior 7 days (shift(1))
rolling_14d_mean : float64         — mean of admissions over prior 14 days
rolling_7d_std   : float64         — std of admissions over prior 7 days

School term feature
-------------------
Deferred. No verified MOEST term dates for 2024–2026 and no investigation
confirming school terms drive KSH inpatient volume (Inv 67 attributes May–Jun
drop to long rains + birth seasonality, not school schedule).
"""

import pandas as pd
try:
    import holidays as hol
    _HOLIDAYS_AVAILABLE = True
except ImportError:
    _HOLIDAYS_AVAILABLE = False

# Oct 4–16 2025: EMR sync failure (Inv 32, Inv 65).
# All gap days are absent from Snowflake — no rows in extractor output.
# Feature builder inserts gap rows with admissions=NaN + is_gap_flagged=True.
_GAP_START = pd.Timestamp("2025-10-04")
_GAP_END   = pd.Timestamp("2025-10-16")

# Smallest lag window — rows before this are all-NaN on lag features.
_LAG_MAX = 28


def build(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full daily feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Output of extractor.pull(). Sparse — one row per day with >= 1 admission.
        Required columns: admission_date (datetime64), admissions (int64), is_gap_flagged (bool).

    Returns
    -------
    pd.DataFrame
        Complete daily rows from df.admission_date.min() to df.admission_date.max().
        Non-gap absent dates have admissions=0.0 (legitimate zero-admission days).
        Gap rows (Oct 4–16) have admissions=NaN, is_gap_flagged=True.
        First _LAG_MAX rows have NaN lag features — trainer must drop.
        Gap-adjacent rows (~40 days) have NaN lag/rolling features — trainer must drop.
    """
    _validate_input(df)

    spine  = _build_spine(df["admission_date"].min(), df["admission_date"].max())
    out    = _merge_actuals(spine, df)
    out    = _apply_gap_flags(out)
    out    = _fill_zeros(out)
    out    = _add_calendar_features(out)
    out    = _add_holiday_flags(out)
    out    = _add_lag_features(out)
    out    = _add_rolling_features(out)

    _validate_output(out)
    return out


def prepare_prophet_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a (ds, y) DataFrame ready for Prophet.

    Excludes gap rows (is_gap_flagged=True) — Prophet handles missing dates natively.
    Includes legitimate zero-admission days (y=0.0) — these are real observed zeros.

    Columns: ds (datetime64), y (float64).
    """
    feat = build(df)
    prophet_df = (
        feat
        .loc[~feat["is_gap_flagged"]]
        [["admission_date", "admissions"]]
        .rename(columns={"admission_date": "ds", "admissions": "y"})
        .reset_index(drop=True)
    )
    return prophet_df


def make_prophet_holidays(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Return a Prophet-compatible holidays DataFrame for Kenya.

    Year range is derived from (start, end) — no hardcoding.
    Returns None if the `holidays` library is unavailable.

    Parameters
    ----------
    start, end : pd.Timestamp — data date range (inclusive)

    Parameters
    ----------
    start, end : pd.Timestamp — data date range (inclusive)

    Holiday window rationale (confirmed 2026-06-23, tune_prophet.py):
      Easter Monday upper_window=4: captures post-Easter suppression through Thu.
        WMAPE improved 51.91→51.45%, coverage 83.3→85.0% on 60-day holdout.
        Caveat: trained on one Easter event (Apr 2025). Effect estimate is noisy.
      All other holidays: upper_window=1 (single-day effect).
    """
    if not _HOLIDAYS_AVAILABLE:
        return None

    _EXTENDED_UPPER = {
        "Easter Monday": 4,   # post-Easter crash confirmed in diagnose_prophet.py
    }

    rows = []
    for year in range(start.year, end.year + 1):
        ke = hol.Kenya(years=year)
        for date, name in ke.items():
            rows.append({
                "ds":           pd.Timestamp(date),
                "holiday":      name,
                "lower_window": 0,
                "upper_window": _EXTENDED_UPPER.get(name, 1),
            })

    return pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _validate_input(df: pd.DataFrame) -> None:
    required = {"admission_date", "admissions", "is_gap_flagged"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"features: missing input columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df["admission_date"]):
        raise ValueError(
            f"features: admission_date must be a datetime dtype, got {df['admission_date'].dtype}"
        )

    if df["admission_date"].duplicated().any():
        n = df["admission_date"].duplicated().sum()
        raise ValueError(
            f"features: {n} duplicate dates in input — extractor GROUP BY may have failed."
        )

    if len(df) < 1:
        raise ValueError("features: empty input DataFrame")


def _build_spine(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(start=start, end=end, freq="D")
    return pd.DataFrame({"admission_date": dates})


def _merge_actuals(spine: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    # Left join — absent dates get admissions=NaN initially.
    # NaN is resolved after gap flags are applied (_fill_zeros).
    merged = spine.merge(
        df[["admission_date", "admissions"]].assign(
            admissions=lambda d: d["admissions"].astype("float64")
        ),
        on="admission_date",
        how="left",
    )
    merged["is_gap_flagged"] = False
    return merged


def _apply_gap_flags(df: pd.DataFrame) -> pd.DataFrame:
    # Oct 4–16 2025: confirmed EMR sync failure. Mark as gap.
    # admissions stays NaN from the left join — gap rows excluded from training.
    gap_mask = (df["admission_date"] >= _GAP_START) & (df["admission_date"] <= _GAP_END)
    df.loc[gap_mask, "is_gap_flagged"] = True
    return df


def _fill_zeros(df: pd.DataFrame) -> pd.DataFrame:
    # Non-gap absent dates → genuine zero-admission days → fill with 0.0.
    # Gap dates retain NaN — data was unavailable, not zero.
    non_gap_nan = df["admissions"].isna() & ~df["is_gap_flagged"]
    df.loc[non_gap_nan, "admissions"] = 0.0
    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"]  = df["admission_date"].dt.dayofweek.astype("int8")
    df["week_of_year"] = df["admission_date"].dt.isocalendar().week.astype("int8")
    df["month"]        = df["admission_date"].dt.month.astype("int8")
    return df


def _add_holiday_flags(df: pd.DataFrame) -> pd.DataFrame:
    if not _HOLIDAYS_AVAILABLE:
        df["is_kenya_holiday"] = False
        return df

    ke_holidays = set()
    for year in df["admission_date"].dt.year.unique():
        ke_holidays.update(hol.Kenya(years=int(year)).keys())

    df["is_kenya_holiday"] = df["admission_date"].dt.date.isin(ke_holidays)
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    # shift(n): lag_n[D] = admissions[D-n]. No future data.
    # Gap days have admissions=NaN — NaN propagates into lags for n days after each gap.
    s = df["admissions"]
    df["lag_7"]  = s.shift(7)
    df["lag_14"] = s.shift(14)
    df["lag_28"] = s.shift(28)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    # shift(1) before rolling: window is [D-n, D-1], not [D-n+1, D].
    # Prevents the current day's admissions from leaking into its own features.
    # min_periods=1 for means: allows partial windows near start without all-NaN.
    # min_periods=2 for std: need at least 2 values; first row is NaN regardless.
    shifted = df["admissions"].shift(1)
    df["rolling_7d_mean"]  = shifted.rolling(7,  min_periods=1).mean()
    df["rolling_14d_mean"] = shifted.rolling(14, min_periods=1).mean()
    df["rolling_7d_std"]   = shifted.rolling(7,  min_periods=2).std()
    return df


def _validate_output(df: pd.DataFrame) -> None:
    # 1. Spine completeness: every calendar day must be present.
    expected_rows = (df["admission_date"].max() - df["admission_date"].min()).days + 1
    if len(df) != expected_rows:
        raise ValueError(
            f"features: spine has {len(df)} rows, expected {expected_rows}. "
            "Duplicate or missing dates in spine."
        )

    # 2. Gap flag count: only validate if the spine intersects the gap window.
    spine_start = df["admission_date"].min()
    spine_end   = df["admission_date"].max()
    if spine_start <= _GAP_END and spine_end >= _GAP_START:
        actual_gap_start = max(spine_start, _GAP_START)
        actual_gap_end   = min(spine_end,   _GAP_END)
        expected_gap     = (actual_gap_end - actual_gap_start).days + 1
        gap_count        = df["is_gap_flagged"].sum()
        if gap_count != expected_gap:
            raise ValueError(
                f"features: expected {expected_gap} gap-flagged rows "
                f"({actual_gap_start.date()} to {actual_gap_end.date()}), "
                f"found {gap_count}."
            )

    # 3. Non-gap rows must have no NaN admissions (zeros were filled).
    non_gap_nan = df.loc[~df["is_gap_flagged"], "admissions"].isna().sum()
    if non_gap_nan > 0:
        raise ValueError(
            f"features: {non_gap_nan} non-gap rows still have NaN admissions — "
            "_fill_zeros may have failed."
        )

    # 4. Calendar features: no NaN allowed.
    cal_cols = ["day_of_week", "week_of_year", "month", "is_kenya_holiday"]
    for col in cal_cols:
        if df[col].isna().any():
            raise ValueError(f"features: NaN found in calendar column '{col}'")

    # 5. Lag NaN count sanity: expect NaN only in first _LAG_MAX rows + gap-adjacent rows.
    #    Post-lag, non-gap rows should have lag_28 populated (warn, don't hard-fail).
    post_lag = df.loc[df.index >= _LAG_MAX]
    post_lag_non_gap = post_lag.loc[~post_lag["is_gap_flagged"]]
    unexpected_lag_nan = post_lag_non_gap["lag_28"].isna().sum()
    if unexpected_lag_nan > _GAP_END.day - _GAP_START.day + _LAG_MAX:
        # More NaN than the gap window can explain — surface as warning, not hard fail.
        print(
            f"features [WARN]: {unexpected_lag_nan} post-lag non-gap rows have NaN lag_28 "
            f"(expected <= gap propagation of ~{_LAG_MAX} rows). Investigate if unexpected."
        )
