"""
predictive.py
-------------
scikit-learn powered analytics for the revenue dashboard.

Each function takes a tidy DataFrame and returns a tidy DataFrame, so the
Streamlit app stays declarative.

Models used:
    - 90-day forecast        : Ridge regression over engineered calendar +
                               lag features, with prediction intervals.
    - Anomaly detection      : Isolation Forest over (date, revenue, residual).
    - Patient segmentation   : KMeans (k=5) over RFM features (z-scored).
    - Churn risk             : logistic-style score from recency vs frequency.
    - Driver importance      : RandomForest feature importance for daily revenue.
    - What-if scenarios      : closed-form sensitivity table (no model).
"""



import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ─── Forecast ────────────────────────────────────────────────────────────────

def _engineer_features(df: pd.DataFrame, date_col: str = "revenue_date") -> pd.DataFrame:
    """Add calendar + lag features for a daily series."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).reset_index(drop=True)

    out["day_of_week"]   = out[date_col].dt.dayofweek
    out["day_of_month"]  = out[date_col].dt.day
    out["month"]         = out[date_col].dt.month
    out["quarter"]       = out[date_col].dt.quarter
    out["week_of_year"]  = out[date_col].dt.isocalendar().week.astype(int)
    out["is_weekend"]    = (out["day_of_week"] >= 5).astype(int)
    out["is_month_end"]  = out[date_col].dt.is_month_end.astype(int)
    out["t"]             = (out[date_col] - out[date_col].min()).dt.days

    out["sin_dow"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["cos_dow"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["sin_doy"] = np.sin(2 * np.pi * out[date_col].dt.dayofyear / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * out[date_col].dt.dayofyear / 365.25)
    return out


def forecast_revenue(
    daily: pd.DataFrame,
    horizon_days: int = 90,
    target_col: str = "revenue",
    date_col: str = "revenue_date",
) -> pd.DataFrame:
    """
    Train Ridge regression on calendar + lag features and forecast forward.

    Returns a DataFrame with columns:
        date, actual, predicted, lower_80, upper_80, segment ('history' | 'forecast')
    """
    df = _engineer_features(daily.rename(columns={target_col: "y"}), date_col=date_col)
    # Lag features (1, 7, 14, 28)
    for lag in (1, 7, 14, 28):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "t", "day_of_week", "day_of_month", "month", "quarter",
        "week_of_year", "is_weekend", "is_month_end",
        "sin_dow", "cos_dow", "sin_doy", "cos_doy",
        "lag_1", "lag_7", "lag_14", "lag_28",
    ]

    X, y = df[feature_cols].values, df["y"].values
    model = Ridge(alpha=2.0, random_state=42)
    model.fit(X, y)

    # Residual std for prediction interval
    in_sample = model.predict(X)
    resid_std = np.std(y - in_sample)

    # Build the future frame
    last_date = df[date_col].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    series = df.copy()
    forecasts = []
    history_y = list(series["y"].values)
    history_dates = list(series[date_col].values)

    for d in future_dates:
        row = {date_col: d}
        row["t"] = (pd.Timestamp(d) - df[date_col].min()).days
        row["day_of_week"]   = d.dayofweek
        row["day_of_month"]  = d.day
        row["month"]         = d.month
        row["quarter"]       = (d.month - 1) // 3 + 1
        row["week_of_year"]  = int(d.isocalendar().week)
        row["is_weekend"]    = int(d.dayofweek >= 5)
        row["is_month_end"]  = int(d.is_month_end)
        row["sin_dow"]       = np.sin(2 * np.pi * d.dayofweek / 7)
        row["cos_dow"]       = np.cos(2 * np.pi * d.dayofweek / 7)
        row["sin_doy"]       = np.sin(2 * np.pi * d.dayofyear / 365.25)
        row["cos_doy"]       = np.cos(2 * np.pi * d.dayofyear / 365.25)
        row["lag_1"]  = history_y[-1]
        row["lag_7"]  = history_y[-7]
        row["lag_14"] = history_y[-14]
        row["lag_28"] = history_y[-28]
        x_pred = np.array([[row[c] for c in feature_cols]])
        yhat = float(model.predict(x_pred)[0])
        row["yhat"] = yhat
        forecasts.append(row)
        history_y.append(yhat)
        history_dates.append(d)

    fc_df = pd.DataFrame(forecasts)

    # Combine history + forecast for plotting
    hist = pd.DataFrame({
        "date":     pd.to_datetime(df[date_col]),
        "actual":   df["y"].values,
        "predicted": in_sample,
        "lower_80": in_sample - 1.282 * resid_std,
        "upper_80": in_sample + 1.282 * resid_std,
        "segment":  "history",
    })
    fc = pd.DataFrame({
        "date":     pd.to_datetime(fc_df[date_col]),
        "actual":   np.nan,
        "predicted": fc_df["yhat"].values,
        "lower_80": fc_df["yhat"].values - 1.282 * resid_std,
        "upper_80": fc_df["yhat"].values + 1.282 * resid_std,
        "segment":  "forecast",
    })
    return pd.concat([hist, fc], ignore_index=True)


# ─── Anomaly detection ───────────────────────────────────────────────────────

def detect_anomalies(
    daily: pd.DataFrame,
    target_col: str = "revenue",
    date_col: str = "revenue_date",
    contamination: float = 0.03,
) -> pd.DataFrame:
    """
    Fit Isolation Forest on (revenue, residual) and tag outliers.

    Returns the input DataFrame with two extra columns: `is_anomaly` (bool)
    and `score` (float — lower is more anomalous).
    """
    df = daily.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # 28-day rolling baseline residual
    df["roll"]     = df[target_col].rolling(28, min_periods=7).median()
    df["residual"] = df[target_col] - df["roll"]
    df["residual"] = df["residual"].fillna(0)

    feats = df[[target_col, "residual"]].values
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(feats)
    df["score"]      = iso.decision_function(feats)
    df["is_anomaly"] = iso.predict(feats) == -1
    return df


# ─── Patient segmentation (RFM) ─────────────────────────────────────────────

def segment_patients(rfm: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    KMeans clustering on RFM. Adds `cluster` and human-readable `segment` labels.
    """
    df = rfm.copy()
    feats = df[["recency_days", "frequency", "monetary"]].values
    feats = np.log1p(np.maximum(feats, 0))
    feats = StandardScaler().fit_transform(feats)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(feats)

    # Label clusters by their economic value
    summary = df.groupby("cluster").agg(
        recency=("recency_days", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
    ).reset_index()
    summary = summary.sort_values("monetary", ascending=False).reset_index(drop=True)

    label_pool = ["Champions", "Loyal", "Promising", "At Risk", "Hibernating"]
    label_map = {row["cluster"]: label_pool[i] if i < len(label_pool) else f"Seg {i+1}"
                 for i, row in summary.iterrows()}
    df["segment"] = df["cluster"].map(label_map)
    return df


# ─── Churn risk score ───────────────────────────────────────────────────────

def churn_risk(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight churn score: combination of recency penalty + frequency boost,
    normalised to 0-100 (higher = more at risk).
    """
    df = rfm.copy()
    rec = (df["recency_days"] - df["recency_days"].min()) / (
        df["recency_days"].max() - df["recency_days"].min() + 1e-9
    )
    freq = (df["frequency"] - df["frequency"].min()) / (
        df["frequency"].max() - df["frequency"].min() + 1e-9
    )
    mon = (df["monetary"] - df["monetary"].min()) / (
        df["monetary"].max() - df["monetary"].min() + 1e-9
    )
    score = 70 * rec - 25 * freq - 5 * mon + 30          # offset
    df["churn_risk"] = np.clip(score, 0, 100).round(1)
    df["risk_band"] = pd.cut(
        df["churn_risk"],
        bins=[-1, 25, 55, 80, 101],
        labels=["Low", "Medium", "High", "Critical"],
    )
    return df


# ─── Driver importance ──────────────────────────────────────────────────────

def revenue_drivers(daily: pd.DataFrame, target_col: str = "revenue",
                    date_col: str = "revenue_date") -> pd.DataFrame:
    """
    Train a RandomForest on calendar features and return importances.
    Returns DataFrame: feature, importance, label.
    """
    df = _engineer_features(daily.rename(columns={target_col: "y"}), date_col=date_col)
    feature_cols = [
        "day_of_week", "day_of_month", "month", "quarter",
        "is_weekend", "is_month_end",
        "sin_dow", "cos_dow", "sin_doy", "cos_doy",
    ]
    rf = RandomForestRegressor(n_estimators=180, random_state=42, max_depth=8, n_jobs=-1)
    rf.fit(df[feature_cols].values, df["y"].values)
    pretty = {
        "day_of_week":   "Weekday vs weekend pattern",
        "day_of_month":  "Day of month (billing cycles)",
        "month":         "Calendar month",
        "quarter":       "Quarter of year",
        "is_weekend":    "Weekend flag",
        "is_month_end":  "Month-end flag",
        "sin_dow":       "Weekly cycle (sin)",
        "cos_dow":       "Weekly cycle (cos)",
        "sin_doy":       "Annual cycle (sin)",
        "cos_doy":       "Annual cycle (cos)",
    }
    out = pd.DataFrame({
        "feature":    feature_cols,
        "importance": rf.feature_importances_,
        "label":      [pretty[c] for c in feature_cols],
    }).sort_values("importance", ascending=False)
    return out


# ─── What-if scenarios (sensitivity table) ──────────────────────────────────

def whatif_scenarios(
    base_daily_revenue: float,
    horizon_days: int = 90,
    levers: dict | None = None,
) -> pd.DataFrame:
    """
    Combine % deltas across levers and report the projected revenue uplift
    for the horizon. No model — pure leverage arithmetic.
    """
    if levers is None:
        levers = {
            "Reduce no-shows by 20%":             0.045,
            "Lift ARPV via cross-sell (Pharmacy)": 0.060,
            "Cut M-Pesa downtime to <0.1%":        0.012,
            "Recover 50% of 90+ day AR":           0.030,
            "Open Eldoret Sat half-day":           0.018,
            "Insurance pre-auth turnaround -1d":   0.022,
            "Tighten discount policy (-30%)":      0.014,
        }
    horizon_rev = base_daily_revenue * horizon_days
    rows = []
    cumulative = horizon_rev
    for name, pct in levers.items():
        uplift = horizon_rev * pct
        cumulative += uplift
        rows.append({
            "lever":        name,
            "uplift_pct":   pct * 100,
            "uplift_value": round(uplift, 2),
            "cumulative":   round(cumulative, 2),
        })
    return pd.DataFrame(rows)

