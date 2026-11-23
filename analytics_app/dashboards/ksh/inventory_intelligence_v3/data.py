"""The only module that runs SQL.

All reads are from HOSPITALS.REPORTING V3 objects (marts, dims, the daily/dispensing fact tables).
Everything loads once in parallel and is cached for an hour; filters are applied in pandas so changing a
filter never re-queries Snowflake.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from . import db

R = "HOSPITALS.REPORTING"

SQL = {
    "stock": f"""
        SELECT as_of_date, store_id, store_name, store_type, facility_name, product_id, product_name,
               product_class, unit_std, therapeutic_class, therapeutic_subclass, clinical_tier, is_equipment,
               qty_on_hand, unit_cost, unit_cost_source, stock_value, consumption_90d, dispensing_90d,
               ward_use_90d, adc_90d, days_of_stock, stock_status, last_consumed_date,
               days_since_last_consumed, priority_action, priority_score
        FROM {R}.RPT_KSH_V3_INV_STOCK_STATUS""",
    "stockouts": f"""
        SELECT store_id, store_name, store_type, product_id, product_name, product_class,
               therapeutic_class, clinical_tier, episode_no, episodes_for_item, stockout_start_date,
               restock_date, is_ongoing, days_out, restocked_by, started_on_system_adjustment
        FROM {R}.RPT_KSH_V3_INV_STOCKOUTS""",
    "control": f"""
        SELECT month, store_id, store_name, store_type, metric, n_events, qty, value_kes
        FROM {R}.RPT_KSH_V3_INV_CONTROL""",
    "daily": f"""
        SELECT f.movement_date, f.store_id, f.product_id, f.net_dispensing_qty, f.net_consumable_qty,
               f.receipt_qty, f.transfer_out_qty, f.day_closing
        FROM {R}.FACT_KSH_V3_INV_DAILY f
        JOIN {R}.DIM_KSH_V3_INV_STORE s ON s.store_id = f.store_id AND s.is_active_store""",
    "products": f"""
        SELECT product_id, product_name, product_class, category_name, unit_std, therapeutic_class,
               therapeutic_subclass, clinical_tier, inn_name, is_consumable
        FROM {R}.DIM_KSH_V3_INV_PRODUCT""",
    "stores": f"""
        SELECT store_id, store_name, store_type, is_active_store, facility_name, parent_store_id
        FROM {R}.DIM_KSH_V3_INV_STORE""",
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_all() -> dict[str, pd.DataFrame]:
    d = db.query_many(SQL)
    for name in ("stock", "stockouts", "daily"):
        for col in d[name].columns:
            if col.endswith("_date") or col == "movement_date":
                d[name][col] = pd.to_datetime(d[name][col], errors="coerce")
    d["control"]["month"] = pd.to_datetime(d["control"]["month"], errors="coerce")
    d["daily"]["consumption"] = d["daily"]["net_dispensing_qty"] + d["daily"]["net_consumable_qty"]
    return d


def as_of(d: dict) -> pd.Timestamp:
    """Stock date: the latest movement date (current stock is as of this day)."""
    return pd.Timestamp(d["stock"]["as_of_date"].max())


def usage_end(d: dict) -> pd.Timestamp:
    """Last COMPLETE day of usage: the extract's final day is partial, so every usage,
    trend and forecast window ends the day before the stock date — the same rule as the marts."""
    return as_of(d) - pd.Timedelta(days=1)


# ── Filters ──────────────────────────────────────────────────────────────────
@dataclass
class Filters:
    stores: list = field(default_factory=list)          # store_id; empty = all active stores
    classes: list = field(default_factory=list)         # product_class
    ther_classes: list = field(default_factory=list)    # therapeutic_class
    tiers: list = field(default_factory=list)           # clinical_tier
    search: str = ""
    stream: str = "All consumption"                     # | "Dispensing only" | "Ward use only"

    def describe(self, store_names: dict) -> list[str]:
        chips = []
        if self.stores:
            chips.append("Stores: " + ", ".join(store_names.get(s, str(s)) for s in self.stores))
        if self.classes:
            chips.append("Class: " + ", ".join(self.classes))
        if self.ther_classes:
            chips.append("Therapeutic: " + ", ".join(self.ther_classes))
        if self.tiers:
            chips.append("Tier: " + ", ".join(self.tiers))
        if self.search:
            chips.append(f'Search: "{self.search}"')
        if self.stream != "All consumption":
            chips.append(f"Demand view: {self.stream}")
        return chips


def product_mask(df: pd.DataFrame, f: Filters, products: pd.DataFrame | None = None) -> pd.Series:
    """Rows whose product passes the product filters. Uses df's own attribute columns when present."""
    src = df
    if products is not None and "product_class" not in df.columns:
        src = df[["product_id"]].merge(products, on="product_id", how="left")
        src.index = df.index
    m = pd.Series(True, index=df.index)
    if f.classes:
        m &= src["product_class"].isin(f.classes)
    if f.ther_classes and "therapeutic_class" in src:
        m &= src["therapeutic_class"].isin(f.ther_classes)
    if f.tiers and "clinical_tier" in src:
        m &= src["clinical_tier"].isin(f.tiers)
    if f.search:
        # brand OR generic name; the catalogue's spellings vary (e.g. AMOXYCILLIN), so the generic helps
        hit = src["product_name"].fillna("").str.contains(f.search, case=False, regex=False)
        if products is not None and "inn_name" in products:
            gen = products.loc[products["inn_name"].fillna("").str.contains(f.search, case=False, regex=False),
                               "product_id"]
            hit |= src["product_id"].isin(gen)
        m &= hit
    return m


def store_mask(df: pd.DataFrame, f: Filters) -> pd.Series:
    if not f.stores or "store_id" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["store_id"].isin(f.stores)


def filtered(df: pd.DataFrame, f: Filters, products: pd.DataFrame | None = None) -> pd.DataFrame:
    return df[store_mask(df, f) & product_mask(df, f, products)]


def stream_column(f: Filters) -> str:
    return {"Dispensing only": "net_dispensing_qty", "Ward use only": "net_consumable_qty"}.get(
        f.stream, "consumption")
