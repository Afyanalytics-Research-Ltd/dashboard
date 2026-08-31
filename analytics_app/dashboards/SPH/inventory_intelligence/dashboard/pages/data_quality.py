"""Data quality & trust — how far to trust what this dashboard shows.

Not a business question but the honesty layer behind the other five pages: which
fields are reliable, which are used with care, and which analyses are held back
because the data can't defend them yet. Forecast accuracy lives here too — as a
trust check, not a headline metric.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from inventory_intelligence.dashboard import charts, components, data_access, theme  # noqa: E402

components.page_header(
    "Data quality & trust",
    "How far to trust what this dashboard shows",
)

# ── Live quality signals ──────────────────────────────────────────────────────

health = data_access.load_table("inventory_health")
forecast = data_access.load_table("demand_forecast")
n_items = int(len(health)) if health is not None else 0
unpriced_share = float("nan")
if health is not None and "unit_price" in health.columns and n_items:
    unpriced_share = float(pd.to_numeric(health["unit_price"], errors="coerce").isna().mean())
censored_share = float("nan")
if forecast is not None and "censored_frac" in forecast.columns:
    cf = pd.to_numeric(forecast["censored_frac"], errors="coerce")
    censored_share = float((cf > 0).mean())
as_of = data_access.soh_as_of()

components.kpi_row(
    [
        {"label": "Items tracked", "value": theme.fmt_compact(n_items),
         "detail": "priced and unpriced together"},
        {"label": "Items with no cost on record", "value": theme.fmt_pct(unpriced_share, 0),
         "detail": "their capital value is understated"},
        {"label": "Items with stock-out gaps", "value": theme.fmt_pct(censored_share, 0),
         "detail": "usage under-recorded on days stock was out"},
        {"label": "Stock snapshot as of",
         "value": data_access.pretty_date(as_of) if as_of else "—",
         "detail": "a point-in-time count, not a live feed"},
    ]
)

# ── Trust ledger ──────────────────────────────────────────────────────────────

components.section_header("What to trust, and what we hold back")
st.caption(
    "Every headline elsewhere in the dashboard rests on these fields. Where a metric "
    "can't be defended, it is flagged here rather than shown with false precision."
)

_up = theme.fmt_pct(unpriced_share, 0) if pd.notna(unpriced_share) else "some"
_cs = theme.fmt_pct(censored_share, 0) if pd.notna(censored_share) else "some"
ledger = pd.DataFrame([
    {"Area": "Capital / inventory value", "Status": "Reliable where priced",
     "What it means": f"{_up} of items have no cost on record, so their value — and the "
     "capital figures that include them — is understated, not wrong."},
    {"Area": "Stock on hand", "Status": "Point-in-time",
     "What it means": "A single counted snapshot, not a live feed. Very high months-of-"
     "supply (>2 years) is quarantined as a probable miscount, not booked as releasable."},
    {"Area": "Consumption / demand", "Status": "Use with care",
     "What it means": f"Store-issue grain (v1). Usage reads as zero on stock-out days, so "
     f"for the {_cs} of items with gaps, demand is under-counted and surplus over-stated."},
    {"Area": "Product identity", "Status": "Not standardized",
     "What it means": "~48 clinical products appear under multiple brand IDs (~21% of the "
     "catalog). This fragments demand and can inflate dead-stock; pooling is a known "
     "follow-up."},
    {"Area": "Margins", "Status": "Not shown",
     "What it means": "Selling price (per unit dispensed) and cost (per pack bought) are on "
     "different unit bases — the raw calc shows a −44% median margin, an artifact. Needs "
     "unit-normalized cost before it can be trusted."},
    {"Area": "Fulfilment (prescription → dispensing)", "Status": "Deferred",
     "What it means": "The v2 data supports it, but the blended rate is misleading: most "
     "un-dispensed lines are discharge/theatre/outpatient scripts. Inpatient dispensing is "
     "~99.5%. Held as a distinct service lens until the population is validated."},
    {"Area": "Two data eras", "Status": "Different timelines",
     "What it means": "Inventory (v1) runs through Jan 2025; prescriptions (v2) Feb 2025 "
     "onward. Fulfilment and inventory metrics can't be joined period-for-period."},
])
st.dataframe(
    ledger, use_container_width=True, hide_index=True,
    column_config={
        "Area": st.column_config.TextColumn("Area", width="medium"),
        "Status": st.column_config.TextColumn("Status", width="small"),
        "What it means": st.column_config.TextColumn("What it means", width="large"),
    },
)
st.caption(
    "**Held back for now:** *margin* analysis (selling and cost sit on different unit "
    "bases) and prescription *fulfilment* (v2 population needs validation). Both are "
    "deferred deliberately rather than shown with numbers we can't defend."
)

# ── Forecast accuracy — a trust check ─────────────────────────────────────────

components.section_header("Do the forecasts match what actually happened?")
validation = data_access.load_table("forecast_validation")
if validation is None or validation.empty:
    components.empty_state("The forecast accuracy check hasn't been run yet.")
else:
    data = validation.copy()
    for col in ["actual", "forecast_low", "forecast_expected", "forecast_high", "abs_pct_error"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    within_rate = data["within_range"].astype(bool).mean() if "within_range" in data.columns else float("nan")
    typical_error = data["abs_pct_error"].median() if "abs_pct_error" in data.columns else float("nan")
    components.kpi_row(
        [
            {"label": "Items checked", "value": theme.fmt_compact(len(data))},
            {"label": "Landed within predicted range", "value": theme.fmt_pct(within_rate),
             "detail": "we aim for around 90%"},
            {"label": "Typical error", "value": theme.fmt_pct(typical_error),
             "detail": "how far the expected figure was off"},
        ]
    )
    ratio = (data["actual"] / data["forecast_expected"].replace(0, pd.NA)).median() \
        if {"actual", "forecast_expected"} <= set(data.columns) else float("nan")
    if pd.notna(ratio) and (ratio < 0.6 or ratio > 1.7):
        direction = "less" if ratio < 1 else "more"
        factor = (1 / ratio) if ratio < 1 else ratio
        st.info(
            f"Actual use ran about **{factor:.1f}× {direction}** than the forecasts expected "
            "— the old ledger (v1) counts stock issued from the store, the new system counts "
            "what's handed to each patient. Expected until the forecasts are recalibrated on "
            "the new data; that is exactly what this check is for.",
            icon=":material/info:",
        )
    named = data_access.with_names(data)
    st.plotly_chart(charts.forecast_vs_actual(named), use_container_width=True)
    st.caption("Each point is one item: predicted along the bottom, actually used up the "
               "side. On the dashed line = spot-on; green landed inside the predicted range.")
