"""Ordering plan — what to order now, with the reasoning behind it."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from inventory_intelligence.dashboard import components, data_access  # noqa: E402

components.page_header(
    "Ordering plan",
    "What to order, and how much",
)

if not data_access.analytics_available():
    components.missing_analytics_stop()


policy = data_access.with_names(data_access.load_table("inventory_policy"))
replen = data_access.load_table("replenishment")
stockout = data_access.load_table("stockout_risk")

selected_categories = components.category_sidebar_filter(policy)
policy = data_access.apply_category_filter(policy, selected_categories)

if policy is None or policy.empty:
    components.empty_state(
        "No ordering plan to show yet.",
        note="The plan needs the demand and delivery models to have run.",
    )
    st.stop()

policy = policy.copy()
if replen is not None:
    replen = replen.copy()
    replen["item_key"] = replen["item_key"].astype(str)
    policy = policy.merge(
        replen[["item_key", "interval_mean", "pooled_level"]],
        on="item_key", how="left", suffixes=("", "_replen"),
    )

soh_date = data_access.soh_as_of() or "?"
if stockout is not None:
    _so = stockout.assign(item_key=stockout["item_key"].astype(str)).set_index("item_key")
    policy["soh"] = policy["item_key"].astype(str).map(_so["soh"])
    if "p_stockout_30" in _so.columns:
        policy["p_stockout_30"] = policy["item_key"].astype(str).map(_so["p_stockout_30"])
else:
    policy["soh"] = np.nan

policy["proposed_qty"] = np.ceil(
    (pd.to_numeric(policy["order_up_to"], errors="coerce")
     - pd.to_numeric(policy["soh"], errors="coerce")).clip(lower=0)
)

# Real lead time (order → goods-receipt), per item — blank where no matched pair.
_leads = data_access.item_lead_times(data_access.selected_facility())
policy["lead_time"] = policy["item_key"].astype(str).map(_leads) if _leads else np.nan

# ── Order worksheet ───────────────────────────────────────────────────────────

st.subheader("What to order now")
st.caption(
    "Sorted by **running-out risk**. Proposed quantity tops each item up from stock on "
    f"hand ({data_access.pretty_date(soh_date)}); edit any figure, then download."
)

_sort = "p_stockout_30" if "p_stockout_30" in policy.columns else "proposed_qty"
work = policy.sort_values(_sort, ascending=False)
worksheet_cols = [c for c in [
    "display_name", "product_category", "p_stockout_30", "soh", "reorder_point",
    "order_up_to", "proposed_qty", "lead_time",
] if c in work.columns]

edited = st.data_editor(
    work[worksheet_cols],
    use_container_width=True,
    hide_index=True,
    height=460,
    disabled=[c for c in worksheet_cols if c != "proposed_qty"],
    column_config={
        "display_name": st.column_config.TextColumn("Item"),
        "product_category": st.column_config.TextColumn("Type"),
        "p_stockout_30": components.chance_column(
            "Chance of running out", "Chance of running out within 1 month — why this "
            "order matters. From the Demand & availability page."),
        "soh": st.column_config.NumberColumn("Stock on hand", format="%.0f"),
        "reorder_point": st.column_config.NumberColumn("Order when stock reaches", format="%.0f"),
        "order_up_to": st.column_config.NumberColumn("Top up to", format="%.0f"),
        "proposed_qty": st.column_config.NumberColumn(
            "Order now (editable)", format="%.0f", min_value=0),
        "lead_time": st.column_config.NumberColumn(
            "Lead time (days)", format="%.0f",
            help="Median days from order to delivery for this item, from matched "
                 "purchase-order / receipt records. Blank where none are matched."),
    },
    key="order_worksheet",
)

n_lines = int((pd.to_numeric(edited["proposed_qty"], errors="coerce") > 0).sum())
st.caption(f"{n_lines} items to order.")
export = edited.copy()
export.insert(0, "facility", data_access.selected_facility())
export["stock_as_of"] = soh_date
st.download_button(
    "Download order list (CSV)",
    export.to_csv(index=False).encode("utf-8"),
    file_name=f"sph_order_list_{soh_date}.csv",
    mime="text/csv",
    icon=":material/download:",
)

# ── The reasoning behind each number ──────────────────────────────────────────

with st.expander("How these levels are set (per item)"):
    st.caption(
        "Buffer stock is the cushion above expected use; target service level is how "
        "sure we want to be of not running out; the typical delivery gap and lead time "
        "set how far ahead each order must cover."
    )
    detail_cols = [c for c in [
        "display_name", "safety_stock", "critical_ratio", "interval_mean", "lead_time",
    ] if c in policy.columns]
    st.dataframe(
        policy.sort_values("order_up_to", ascending=False)[detail_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "display_name": st.column_config.TextColumn("Item"),
            "safety_stock": st.column_config.NumberColumn("Buffer stock", format="%.0f"),
            "critical_ratio": st.column_config.NumberColumn(
                "Target service level", format="percent",
                help="How sure we want to be of not running out."),
            "interval_mean": st.column_config.NumberColumn(
                "Typical gap between deliveries (days)", format="%.0f"),
            "lead_time": st.column_config.NumberColumn("Lead time (days)", format="%.0f"),
        },
    )
