"""Demand panel construction.

The forecasting layer consumes a daily item × date grid, not raw dispense
events. Three integrity rules are enforced here:

1. Zero-fill only inside each item's active window — from its first dispense
   to the analysis horizon. Days before an item existed are not zero-demand
   days.
2. Censoring, not zeros, when stock was out: a zero-demand day whose carried
   stock-on-hand is ≤ 0 is a day demand was unobservable. It is flagged
   ``censored=True`` and excluded from model likelihoods. Days with no stock
   signal stay ``censored=<NA>`` — unknown, never guessed.
3. Mask windows (e.g. migration backfill spikes found by the anomaly
   backcast) flag ``masked=True`` and are excluded from every likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

import numpy as np
import pandas as pd

PANEL_COLUMNS = ["item_key", "date", "quantity", "n_events", "censored", "masked"]


@dataclass
class DemandPanel:
    daily: pd.DataFrame
    meta: pd.DataFrame


def _mask_flags(
    dates: pd.DatetimeIndex, mask_windows: Optional[Iterable[tuple[date, date]]]
) -> np.ndarray:
    masked = np.zeros(len(dates), dtype=bool)
    for start, end in mask_windows or []:
        masked |= np.asarray(
            (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        )
    return masked


def build_demand_panel(
    consumption: pd.DataFrame,
    stock: Optional[pd.DataFrame] = None,
    facility_horizon: Optional[date] = None,
    mask_windows: Optional[list[tuple[date, date]]] = None,
) -> DemandPanel:
    """Build the daily demand panel from consumption events.

    Parameters
    ----------
    consumption : output of ``ingestion.load_consumption`` (must include
        ``item_key, dispensed_at, quantity``; ``soh_before`` and
        ``unit_price`` are used when present).
    stock : optional ``ingestion.load_stock_snapshot`` frame; merged into
        ``meta`` (soh, soh_as_of).
    facility_horizon : end of every item's zero-fill window. ``None`` → the
        max dispense date in the data.
    mask_windows : windows to flag ``masked=True`` — from the anomaly
        backcast, never hardcoded.
    """
    df = consumption.copy()
    df["dispensed_at"] = pd.to_datetime(df["dispensed_at"])
    df = df.dropna(subset=["item_key", "dispensed_at"])
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
    df = df.sort_values(["item_key", "dispensed_at"], kind="mergesort")
    df["date"] = df["dispensed_at"].dt.normalize()

    horizon = (
        pd.Timestamp(facility_horizon) if facility_horizon is not None else df["date"].max()
    )

    if "soh_before" in df.columns:
        df["_soh_after_est"] = pd.to_numeric(df["soh_before"], errors="coerce") - df["quantity"]
    else:
        df["_soh_after_est"] = np.nan
    if "unit_price" in df.columns:
        df["_line_value"] = df["quantity"] * pd.to_numeric(df["unit_price"], errors="coerce")
    else:
        df["_line_value"] = np.nan

    daily_events = (
        df.groupby(["item_key", "date"], sort=True)
        .agg(
            quantity=("quantity", "sum"),
            n_events=("quantity", "size"),
            _soh_last=("_soh_after_est", "last"),
            _value=("_line_value", "sum"),
        )
        .reset_index()
    )

    frames: list[pd.DataFrame] = []
    meta_rows: list[dict] = []
    for item_key, g in daily_events.groupby("item_key", sort=True):
        first = g["date"].min()
        grid = pd.date_range(first, max(horizon, first), freq="D")
        g = g.set_index("date").reindex(grid)

        quantity = g["quantity"].fillna(0.0).to_numpy()
        n_events = g["n_events"].fillna(0).astype(int).to_numpy()
        soh_ffill = g["_soh_last"].ffill()
        masked = _mask_flags(grid, mask_windows)

        censored = pd.array([pd.NA] * len(grid), dtype="boolean")
        nonzero = quantity > 0
        censored[nonzero] = False
        zero_known = (~nonzero) & soh_ffill.notna().to_numpy()
        censored[zero_known] = pd.array(
            soh_ffill.to_numpy()[zero_known] <= 0, dtype="boolean"
        )

        frames.append(
            pd.DataFrame(
                {
                    "item_key": item_key,
                    "date": grid,
                    "quantity": quantity,
                    "n_events": n_events,
                    "censored": censored,
                    "masked": masked,
                }
            )
        )

        observable = ~masked
        active = quantity > 0
        meta_rows.append(
            {
                "item_key": item_key,
                "first_activity": first,
                "last_activity": grid[active].max() if active.any() else first,
                "active_days": int(len(grid)),
                "n_events": int(n_events.sum()),
                "total_qty": float(quantity.sum()),
                "value": float(g["_value"].sum(min_count=1)) if g["_value"].notna().any() else np.nan,
                "censored_frac": float(
                    (censored[observable] == True).sum() / max(observable.sum(), 1)  # noqa: E712
                ),
                "masked_days": int(masked.sum()),
            }
        )

    daily = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=PANEL_COLUMNS)
    )
    meta = pd.DataFrame(meta_rows)

    if stock is not None and not meta.empty:
        soh_cols = [c for c in ("item_key", "soh", "soh_raw", "soh_as_of") if c in stock.columns]
        meta = meta.merge(stock[soh_cols].drop_duplicates("item_key"), on="item_key", how="left")

    return DemandPanel(daily=daily, meta=meta)


# ── Calendar features ─────────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Attach day-of-week / month / ISO week-of-year / year columns.

    These are *inputs to fitted models* (harmonic seasonality terms) —
    no weekday/seasonal adjustment constants live here.
    """
    out = df.copy()
    ts = pd.to_datetime(out[date_col])
    out["dow"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    out["year"] = ts.dt.year
    return out


# ── Weekly procedure counts ───────────────────────────────────────────────────

def weekly_procedure_counts(
    procedures: pd.DataFrame, date_col: str = "requested_at"
) -> pd.DataFrame:
    """Theatre procedure counts per ISO week (weeks start Monday)."""
    ts = pd.to_datetime(procedures[date_col]).dropna()
    weeks = ts.dt.to_period("W-SUN").dt.start_time
    counts = weeks.value_counts().sort_index()
    return pd.DataFrame({"week_start": counts.index, "procedure_count": counts.to_numpy()})


def add_weekly_procedure_counts(
    daily: pd.DataFrame,
    procedures: pd.DataFrame,
    date_col: str = "date",
    procedure_date_col: str = "requested_at",
) -> pd.DataFrame:
    """Join weekly theatre volumes onto the daily panel.

    Weeks *inside* theatre-data coverage with no requests are true zeros;
    weeks *outside* coverage (theatre data is v2-only) stay <NA> — absence
    of data is never coded as absence of surgery.
    """
    wk = weekly_procedure_counts(procedures, date_col=procedure_date_col)
    out = daily.copy()
    out["week_start"] = pd.to_datetime(out[date_col]).dt.to_period("W-SUN").dt.start_time
    out = out.merge(wk, on="week_start", how="left")
    if not wk.empty:
        in_coverage = out["week_start"].between(wk["week_start"].min(), wk["week_start"].max())
        out.loc[in_coverage, "procedure_count"] = out.loc[in_coverage, "procedure_count"].fillna(0)
    return out
