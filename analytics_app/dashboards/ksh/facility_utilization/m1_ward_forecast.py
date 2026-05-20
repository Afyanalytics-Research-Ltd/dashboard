"""
M1 Ward Demand Forecast — production module
Holt's Linear Trend with automatic linear trendline fallback.
Called by the Predictive Analytics dashboard page.

Returns get_forecast() -> (df_hist, df_fcast)
  df_hist:  cleaned historical monthly admissions per series
  df_fcast: 3-month forward forecast per series with 90% prediction interval
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import Holt
from snowflake_service.snowflake_client import SnowflakeClient
run_query = SnowflakeClient().query

# ── Constants ──────────────────────────────────────────────────────────────────
MAPE_THRESHOLD   = 15.0
N_AHEAD          = 3
MIN_TRAIN_MONTHS = 10
VALIDATED_DATE   = pd.Timestamp("2026-05-14")
RETRAIN_DATE     = pd.Timestamp("2026-11-14")

TENRI_STARTUP_DROP    = pd.to_datetime(["2018-05-01", "2018-10-01", "2018-11-01"])
TENRI_HOLDOUT_CUTOFF  = pd.Timestamp("2022-04-01")
KSH_HOLDOUT_CUTOFF    = pd.Timestamp("2026-01-01")

TENRI_VIABLE_WARDS = [
    "Surgical", "General", "Maternity",
    "Medical — Male", "Medical — Female", "Paediatric",
]

FAC_CONFIG = {
    "TENRI":        ("TENRI", TENRI_HOLDOUT_CUTOFF),
    "KISUMU_CLEAN": ("KSH",   KSH_HOLDOUT_CUTOFF),
}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _drop_partial_last_month(df, group_col):
    max_m = df.groupby(group_col)["admission_month"].max().rename("_max")
    return (
        df.join(max_m, on=group_col)
          .pipe(lambda d: d[d["admission_month"] < d["_max"]])
          .drop(columns="_max")
          .reset_index(drop=True)
    )


def _fit_series(y_all, y_train, y_holdout, last_month, series, facility, ward):
    """Fit Holt's; fall back to linear trendline if MAPE >= threshold.
    Returns (fcast_df, model_type, mape_val)."""
    n_holdout = len(y_holdout)

    # Holdout validation
    fit_val  = Holt(y_train, initialization_method="estimated").fit(optimized=True)
    y_pred   = fit_val.forecast(n_holdout)
    mape_val = float(np.mean(np.abs((y_holdout - y_pred) / y_holdout)) * 100)

    h             = np.sqrt(np.arange(1, N_AHEAD + 1))
    future_months = [last_month + pd.DateOffset(months=i + 1) for i in range(N_AHEAD)]

    if mape_val < MAPE_THRESHOLD:
        fit_full   = Holt(y_all, initialization_method="estimated").fit(optimized=True)
        pred       = fit_full.forecast(N_AHEAD)
        sigma      = float((y_all - fit_full.fittedvalues).std())
        model_type = "holts"
        mape_out   = round(mape_val, 1)
    else:
        x           = np.arange(len(y_all), dtype=float)
        slope, intercept = np.polyfit(x, y_all, 1)
        x_fut       = np.arange(len(y_all), len(y_all) + N_AHEAD, dtype=float)
        pred        = slope * x_fut + intercept
        sigma       = float((y_all - (slope * x + intercept)).std())
        model_type  = "trendline"
        mape_out    = None

    rows = []
    for i, (m, p) in enumerate(zip(future_months, pred)):
        rows.append({
            "series":         f"{facility} — {ward}",
            "facility":       facility,
            "ward":           ward,
            "forecast_month": m,
            "low_90":         max(0, round(p - 1.64 * sigma * h[i])),
            "point":          round(p),
            "high_90":        round(p + 1.64 * sigma * h[i]),
            "model_type":     model_type,
            "mape":           mape_out,
            "month_offset":   i + 1,
        })

    return pd.DataFrame(rows), model_type, mape_out


# ── Data pulls ─────────────────────────────────────────────────────────────────

def _pull_facility_data():
    sql = """
    SELECT
        source_schema,
        DATE_TRUNC('month', admission_date)::DATE AS admission_month,
        COUNT(*) AS admissions
    FROM HOSPITALS.STAGING.stg_inpatient_admissions
    WHERE source_schema IN ('KISUMU_CLEAN', 'TENRI')
      AND NOT (source_schema = 'TENRI' AND ward_category = 'Specialty')
    GROUP BY source_schema, admission_month
    ORDER BY source_schema, admission_month
    """
    df = run_query(sql)
    df.columns = [c.lower() for c in df.columns]
    df["admission_month"] = pd.to_datetime(df["admission_month"])
    df = _drop_partial_last_month(df, "source_schema")
    df = df[~(
        (df["source_schema"] == "TENRI") &
        (df["admission_month"].isin(TENRI_STARTUP_DROP))
    )].reset_index(drop=True)
    return df


def _pull_ward_data():
    placeholders = ", ".join(f"'{w}'" for w in TENRI_VIABLE_WARDS)
    sql = f"""
    SELECT
        ward_category,
        DATE_TRUNC('month', admission_date)::DATE AS admission_month,
        COUNT(*) AS admissions
    FROM HOSPITALS.STAGING.stg_inpatient_admissions
    WHERE source_schema = 'TENRI'
      AND ward_category IN ({placeholders})
    GROUP BY ward_category, admission_month
    ORDER BY ward_category, admission_month
    """
    df = run_query(sql)
    df.columns = [c.lower() for c in df.columns]
    df["admission_month"] = pd.to_datetime(df["admission_month"])
    df = _drop_partial_last_month(df, "ward_category")
    df = df[~df["admission_month"].isin(TENRI_STARTUP_DROP)].reset_index(drop=True)
    return df


# ── Public API ─────────────────────────────────────────────────────────────────

def get_forecast():
    """
    Returns (df_hist, df_fcast).

    df_hist columns:  series, facility, ward, admission_month, admissions
    df_fcast columns: series, facility, ward, forecast_month,
                      low_90, point, high_90, model_type, mape, month_offset
    """
    df_fac  = _pull_facility_data()
    df_ward = _pull_ward_data()

    hist_frames  = []
    fcast_frames = []

    # ── Facility-level (both facilities) ──────────────────────────────────────
    for schema, (fac_label, holdout_cutoff) in FAC_CONFIG.items():
        s = df_fac[df_fac["source_schema"] == schema].sort_values("admission_month").reset_index(drop=True)

        y_all     = s["admissions"].values.astype(float)
        y_train   = s[s["admission_month"] < holdout_cutoff]["admissions"].values.astype(float)
        y_holdout = s[s["admission_month"] >= holdout_cutoff]["admissions"].values.astype(float)
        last_m    = s["admission_month"].max()

        fdf, _, _ = _fit_series(y_all, y_train, y_holdout, last_m, fac_label, fac_label, "Facility")
        fcast_frames.append(fdf)

        hdf = s[["admission_month", "admissions"]].copy()
        hdf["series"]   = f"{fac_label} — Facility"
        hdf["facility"] = fac_label
        hdf["ward"]     = "Facility"
        hist_frames.append(hdf)

    # ── TENRI ward-level ──────────────────────────────────────────────────────
    for ward, grp in df_ward.groupby("ward_category"):
        grp = grp.sort_values("admission_month").reset_index(drop=True)
        y_all     = grp["admissions"].values.astype(float)
        y_train   = grp[grp["admission_month"] < TENRI_HOLDOUT_CUTOFF]["admissions"].values.astype(float)
        y_holdout = grp[grp["admission_month"] >= TENRI_HOLDOUT_CUTOFF]["admissions"].values.astype(float)
        last_m    = grp["admission_month"].max()

        if len(y_train) < MIN_TRAIN_MONTHS or len(y_holdout) < 3:
            continue

        fdf, _, _ = _fit_series(y_all, y_train, y_holdout, last_m, "TENRI", "TENRI", ward)
        fcast_frames.append(fdf)

        hdf = grp[["admission_month", "admissions"]].copy()
        hdf["series"]   = f"TENRI — {ward}"
        hdf["facility"] = "TENRI"
        hdf["ward"]     = ward
        hist_frames.append(hdf)

    df_hist  = pd.concat(hist_frames,  ignore_index=True)
    df_fcast = pd.concat(fcast_frames, ignore_index=True)
    return df_hist, df_fcast
