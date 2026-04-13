"""
data/loader.py

PharmaPlus-native simulation engine.
SKU universe = PharmaPlus scraped catalogue (2,488 products).
No database connection required.

Signal tiers per SKU:
    A — is_best_seller=True OR units_sold non-null  → fast/medium mover, tight variance
    B — is_on_promotion=True OR in_stock=False       → moderate velocity, higher variance
    C — no signals                                   → slow mover default, dead stock injected

Inventory qty derived from price bracket:
    qty_on_hand ≈ BASE_UNITS / sqrt(price_kes) * noise
    KSh 50 product → ~100 units | KSh 5,000 product → ~7 units

Branch profiles (demographically grounded):
    1 Karen Hub     — balanced, affluent mixed catchment
    2 Westlands     — pharma + non-pharma dominant, high footfall
    3 Kilimani      — beauty dominant, highest beauty dead stock
    4 Nyali Mombasa — pharma/OTC heavy, weak supplements & bodybuilding
    5 Two Rivers    — supplements + bodybuilding dominant (mall catchment)
"""

import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
import streamlit as st

from engine.catalogue_matcher import (
    load_pharmaplus_catalogue,
    DEAD_STOCK_TIERS_BY_CATEGORY,
    _map_pharmaplus_category,
)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

BRANCH_NAMES = {
    1: "PharmaPlus Karen (Hub)",
    2: "PharmaPlus Westlands",
    3: "PharmaPlus Kilimani",
    4: "PharmaPlus Nyali (Mombasa)",
    5: "PharmaPlus Two Rivers",
}

SIMULATION_DAYS = 180
SEED            = 42
BASE_UNITS      = 500   # base qty for a KSh 1 product (scales down with price)

# ─── VELOCITY BENCHMARKS ─────────────────────────────────────────────────────
# avg daily units by (category, signal_tier); noise_std on top

VELOCITY = {
    "Pharma":                 {"A": {"mean": 4.5, "noise": 1.0}, "B": {"mean": 1.5, "noise": 0.6}, "C": {"mean": 0.2,  "noise": 0.1}},
    "Beauty & Cosmetics":     {"A": {"mean": 2.5, "noise": 0.8}, "B": {"mean": 0.8, "noise": 0.4}, "C": {"mean": 0.1,  "noise": 0.05}},
    "Vitamins & Supplements": {"A": {"mean": 3.0, "noise": 0.9}, "B": {"mean": 1.0, "noise": 0.5}, "C": {"mean": 0.15, "noise": 0.07}},
    "Body Building":          {"A": {"mean": 2.0, "noise": 0.7}, "B": {"mean": 0.7, "noise": 0.3}, "C": {"mean": 0.08, "noise": 0.04}},
    "Non-Pharma":             {"A": {"mean": 3.5, "noise": 1.0}, "B": {"mean": 1.2, "noise": 0.5}, "C": {"mean": 0.18, "noise": 0.08}},
}

# Fallback tier split when no signal available (A, B, C fractions — must sum to 1)
DEFAULT_TIER_SPLIT = {
    "Pharma":                 (0.20, 0.45, 0.35),
    "Beauty & Cosmetics":     (0.12, 0.35, 0.53),
    "Vitamins & Supplements": (0.15, 0.40, 0.45),
    "Body Building":          (0.10, 0.30, 0.60),
    "Non-Pharma":             (0.22, 0.42, 0.36),
}

# Dead stock injection: % of Tier-C SKUs that get frozen beyond DEAD threshold
# Calibrated so ~8–15% of overall SKUs end up as dead stock per branch
DEAD_INJECT = {
    "Pharma":                  {1: 0.08, 2: 0.06, 3: 0.10, 4: 0.06, 5: 0.12},
    "Beauty & Cosmetics":      {1: 0.10, 2: 0.08, 3: 0.15, 4: 0.18, 5: 0.10},
    "Vitamins & Supplements":  {1: 0.09, 2: 0.08, 3: 0.12, 4: 0.15, 5: 0.06},
    "Body Building":           {1: 0.12, 2: 0.10, 3: 0.14, 4: 0.20, 5: 0.05},
    "Non-Pharma":              {1: 0.07, 2: 0.06, 3: 0.09, 4: 0.08, 5: 0.10},
}

# ─── BRANCH PROFILES ─────────────────────────────────────────────────────────
# (demand_multiplier, sku_coverage_fraction) per branch per category

BRANCH_PROFILES = {
    1: {  # Karen Hub — balanced
        "Pharma":                 (1.0, 0.85),
        "Beauty & Cosmetics":     (1.1, 0.75),
        "Vitamins & Supplements": (1.0, 0.70),
        "Body Building":          (0.8, 0.55),
        "Non-Pharma":             (1.0, 0.80),
    },
    2: {  # Westlands — pharma + non-pharma dominant
        "Pharma":                 (1.4, 0.90),
        "Beauty & Cosmetics":     (0.7, 0.50),
        "Vitamins & Supplements": (0.9, 0.60),
        "Body Building":          (0.5, 0.35),
        "Non-Pharma":             (1.5, 0.85),
    },
    3: {  # Kilimani — beauty dominant
        "Pharma":                 (0.8, 0.70),
        "Beauty & Cosmetics":     (1.9, 0.92),
        "Vitamins & Supplements": (1.2, 0.75),
        "Body Building":          (0.6, 0.40),
        "Non-Pharma":             (0.7, 0.55),
    },
    4: {  # Nyali Mombasa — pharma heavy
        "Pharma":                 (1.6, 0.90),
        "Beauty & Cosmetics":     (0.5, 0.40),
        "Vitamins & Supplements": (0.6, 0.45),
        "Body Building":          (0.3, 0.25),
        "Non-Pharma":             (1.2, 0.75),
    },
    5: {  # Two Rivers — supplements + bodybuilding dominant
        "Pharma":                 (0.7, 0.65),
        "Beauty & Cosmetics":     (1.0, 0.65),
        "Vitamins & Supplements": (1.7, 0.88),
        "Body Building":          (2.2, 0.85),
        "Non-Pharma":             (0.8, 0.60),
    },
}

SHELF_LIFE_DAYS = {
    "Pharma":                 730,
    "Beauty & Cosmetics":     1095,
    "Vitamins & Supplements": 730,
    "Body Building":          548,
    "Non-Pharma":             1825,
}


# ─── SIGNAL TIER ASSIGNMENT ───────────────────────────────────────────────────

def _assign_signal_tiers(
    products:    pd.DataFrame,
    goodlife_df: pd.DataFrame,
    linton_df:   pd.DataFrame,
    rng:         np.random.Generator,
) -> pd.Series:
    """
    Assigns each SKU to signal tier A, B, or C based on available signals.
    A = strong demand evidence | B = weak signal | C = no signal
    """
    n     = len(products)
    tiers = pd.Series(["C"] * n, index=products.index)

    # --- Tier A ---
    best_seller = products.get("is_best_seller", pd.Series(False, index=products.index))
    best_seller = pd.to_numeric(best_seller, errors="coerce").fillna(0).astype(bool)

    units_sold = pd.to_numeric(
        products.get("units_sold", pd.Series(np.nan, index=products.index)),
        errors="coerce",
    )
    has_units = units_sold.notna() & (units_sold > 0)

    # Competitor review signal via brand token
    comp_reviews = pd.Series(0.0, index=products.index)
    comp_frames  = [df for df in [goodlife_df, linton_df] if not df.empty]
    if comp_frames:
        comp_all = pd.concat(comp_frames, ignore_index=True)
        if "reviews" in comp_all.columns and "product_name" in comp_all.columns:
            comp_all["brand_tok"] = comp_all["product_name"].str.lower().str.split().str[0].fillna("")
            brand_rev = comp_all.groupby("brand_tok")["reviews"].max().to_dict()
            prod_btok = products["brand"].fillna("").str.lower().str.split().str[0].fillna("")
            comp_reviews = prod_btok.map(brand_rev).fillna(0)

    tier_a = best_seller | has_units | (comp_reviews > 50)
    tiers[tier_a] = "A"

    # --- Tier B ---
    remaining  = tiers == "C"
    on_promo   = pd.to_numeric(
        products.get("is_on_promotion", pd.Series(False, index=products.index)),
        errors="coerce",
    ).fillna(0).astype(bool)

    in_stock_col = products.get("in_stock", pd.Series(1, index=products.index))
    out_of_stock = (pd.to_numeric(in_stock_col, errors="coerce").fillna(1) == 0)
    has_reviews  = comp_reviews > 0

    tier_b = remaining & (on_promo | out_of_stock | has_reviews)
    tiers[tier_b] = "B"

    # --- Tier C probabilistic fill ---
    still_c = tiers == "C"
    if still_c.sum() > 0:
        for cat, (fa, fb, fc) in DEFAULT_TIER_SPLIT.items():
            mask = still_c & (products["internal_category"] == cat)
            if mask.sum() == 0:
                continue
            drawn = rng.choice(["A", "B", "C"], size=mask.sum(), p=[fa, fb, fc])
            tiers[mask] = drawn

    return tiers


# ─── QTY FROM PRICE ───────────────────────────────────────────────────────────

def _qty_from_price(price: float, rng: np.random.Generator) -> int:
    price = max(price, 10.0)
    base  = BASE_UNITS / np.sqrt(price)
    return max(2, int(round(base * rng.uniform(0.4, 2.0))))


# ─── COMPETITOR DATA HELPERS ─────────────────────────────────────────────────

def _load_competitor_reviews(
    goodlife_path: str | None,
    linton_path:   str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = {}
    for path, label in [(goodlife_path, "goodlife"), (linton_path, "linton")]:
        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
                df["source"]  = label
                df["reviews"] = pd.to_numeric(df.get("reviews", pd.Series(dtype=float)), errors="coerce").fillna(0)
                df["rating"]  = pd.to_numeric(df.get("rating",  pd.Series(dtype=float)), errors="coerce").fillna(3.0)
                result[label] = df[["product_name", "reviews", "rating", "source"]]
            except Exception as e:
                st.warning(f"Could not load {label}: {e}")
    return result.get("goodlife", pd.DataFrame()), result.get("linton", pd.DataFrame())


# ─── SIMULATION CORE ─────────────────────────────────────────────────────────

def _simulate_branch(
    products:     pd.DataFrame,
    branch_id:    int,
    signal_tiers: pd.Series,
    rng:          np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates dispensing history and inventory snapshot for one branch.
    Returns (dispensing_df, inventory_df).
    """
    profile  = BRANCH_PROFILES[branch_id]
    end_date = date.today()
    dates    = pd.date_range(end=pd.Timestamp(end_date), periods=SIMULATION_DAYS, freq="D")

    disp_records = []
    inv_records  = []

    for cat, (demand_mult, sku_frac) in profile.items():
        cat_prods = products[products["internal_category"] == cat]
        if cat_prods.empty:
            continue

        n_carry  = max(1, int(len(cat_prods) * sku_frac))
        idx      = rng.choice(cat_prods.index, size=n_carry, replace=False)
        carried  = cat_prods.loc[idx]

        dead_rate   = DEAD_INJECT.get(cat, {}).get(branch_id, 0.10)
        dead_thresh = DEAD_STOCK_TIERS_BY_CATEGORY.get(cat, {}).get("DEAD", 90)
        vel_cfg     = VELOCITY.get(cat, VELOCITY["Pharma"])

        for _, row in carried.iterrows():
            pid       = row["product_id"]
            price     = max(float(row.get("price_kes") or 100), 10.0)
            tier      = signal_tiers.get(row.name, "C")
            cfg       = vel_cfg[tier]
            base_vel  = max(0.01, cfg["mean"] * demand_mult)
            unit_cost = price * rng.uniform(0.55, 0.75)

            # Dead stock injection for Tier-C slow movers
            is_dead = (tier == "C") and (rng.random() < dead_rate)
            if is_dead:
                cutoff = max(0, min(
                    SIMULATION_DAYS - int(rng.uniform(dead_thresh + 5, dead_thresh + 60)),
                    SIMULATION_DAYS - 1,
                ))
                active_dates = dates[:cutoff]
            else:
                active_dates = dates

            # Dispensing events — sparse (only days with sales > 0)
            for dt in active_dates:
                daily = max(0.001, base_vel + rng.normal(0, cfg["noise"]))
                qty   = int(rng.poisson(daily))
                if qty <= 0:
                    continue
                disp_records.append({
                    "date":               dt,
                    "product_id":         pid,
                    "store_id":           branch_id,
                    "qty_dispensed":      qty,
                    "unit_selling_price": round(price, 2),
                    "total_sales_value":  round(qty * price, 2),
                    "unit_cost":          round(unit_cost, 2),
                    "total_cost_value":   round(qty * unit_cost, 2),
                })

            # Inventory snapshot
            if is_dead:
                price_qty = max(6, _qty_from_price(price, rng))
                qty_oh    = max(1, int(rng.uniform(5, price_qty)))
            elif tier == "A":
                qty_oh = max(1, int(base_vel * 14 * rng.uniform(0.5, 1.8)))
            elif tier == "B":
                qty_oh = max(1, int(rng.uniform(3, 30)))
            else:
                qty_oh = max(0, int(rng.uniform(0, 15)))

            reorder = max(1, int(rng.uniform(3, 12)))
            inv_records.append({
                "snapshot_date":         pd.Timestamp(end_date),
                "store_id":              branch_id,
                "product_id":            pid,
                "qty_on_hand":           qty_oh,
                "smart_reorder_level":   reorder,
                "re_order_level":        reorder,
                "unit_cost":             round(unit_cost, 2),
                "total_inventory_value": round(qty_oh * unit_cost, 2),
                "is_stockout":           int(qty_oh <= 0),
                "is_low_stock":          int(0 < qty_oh <= reorder),
                "snapshot_id":           f"{pid}-{branch_id}",
            })

    disp_df = pd.DataFrame(disp_records)
    if not disp_df.empty:
        disp_df["dispensing_id"] = range(
            branch_id * 10_000_000,
            branch_id * 10_000_000 + len(disp_df),
        )

    return disp_df, pd.DataFrame(inv_records)


# ─── MAIN ASSEMBLER ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def build_multi_branch_data(
    pharmaplus_csv_path: str,
    goodlife_csv_path:   str | None = None,
    linton_csv_path:     str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds the full multi-branch dataset from PharmaPlus catalogue alone.

    Returns:
        dispensing_all, inventory_all, products, match_table
    """
    if not pharmaplus_csv_path or not os.path.exists(pharmaplus_csv_path):
        st.error("PharmaPlus catalogue CSV not found. Check the path in the sidebar.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    with st.spinner("Loading PharmaPlus catalogue…"):
        catalogue = load_pharmaplus_catalogue(pharmaplus_csv_path)

    # Build products master from catalogue
    products = catalogue.rename(columns={
        "pharmaplus_sku": "product_id",
        "name":           "product_name",
    }).copy()
    products["category_name"]   = products["internal_category"]
    products["unit_of_measure"] = "unit"
    products["shelf_life_days"] = (
        products["internal_category"].map(SHELF_LIFE_DAYS).fillna(730).astype(int)
    )

    # Load competitor data for signal tier assignment
    goodlife_df, linton_df = _load_competitor_reviews(goodlife_csv_path, linton_csv_path)

    # Assign signal tiers once (shared across all branches)
    master_rng   = np.random.default_rng(SEED)
    signal_tiers = _assign_signal_tiers(products, goodlife_df, linton_df, master_rng)
    products["signal_tier"] = signal_tiers.values

    # Simulate all branches
    disp_parts, inv_parts = [], []
    for branch_id, branch_name in BRANCH_NAMES.items():
        rng = np.random.default_rng(SEED + branch_id * 17)
        with st.spinner(f"Simulating {branch_name}…"):
            disp, inv = _simulate_branch(products, branch_id, signal_tiers, rng)
        if not disp.empty:
            disp_parts.append(disp)
        if not inv.empty:
            inv_parts.append(inv)

    dispensing_all = pd.concat(disp_parts, ignore_index=True) if disp_parts else pd.DataFrame()
    inventory_all  = pd.concat(inv_parts,  ignore_index=True) if inv_parts  else pd.DataFrame()

    # Identity match table — PharmaPlus IS the basket, no translation needed
    match_table = products[["product_id", "product_name", "internal_category", "brand"]].copy()
    match_table["pharmaplus_sku"]        = match_table["product_id"]
    match_table["price_kes_pharmaplus"]  = products.get(
        "price_kes", pd.Series(np.nan, index=products.index)
    ).values
    match_table["match_type"]  = "direct"
    match_table["match_score"] = 100

    return dispensing_all, inventory_all, products, match_table