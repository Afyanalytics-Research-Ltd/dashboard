"""
PharmaPlus Kenya — Competitor Pricing Module
=============================================
Loads scraped competitor CSVs, normalises to a common schema,
fuzzy-matches against PharmaPlus SKUs, and outputs a price
comparison table ready for the engine and dashboard.

Sources:
    data/goodlife_price_list.csv   — 4,207 products
    data/linton.csv                — 816 products (beauty/cosmetics)
    data/mydawa_product_list.csv   — 473 products

Match strategy:
    1. Normalise all names (lowercase, strip punctuation)
    2. Fuzzy match on product_name (token_sort_ratio >= THRESHOLD)
    3. MyDawa: also try brand_name + name; take best score
    4. Post-filter: reject if both names have a size unit and
       they differ by more than SIZE_TOLERANCE

Output: data/competitor_prices.csv
"""

import logging
import os
import re

import polars as pl
from rapidfuzz import fuzz, process

# ── Config ─────────────────────────────────────────────────────────────────────
THRESHOLD      = 80    # Minimum fuzzy match score to accept
SIZE_TOLERANCE = 0.20  # Max allowed size ratio difference (20%)

SOURCES = {
    "goodlife": {
        "path":      "data/goodlife_price_list.csv",
        "name_col":  "product_name",
        "price_col": "current_price",
        "brand_col": None,
    },
    "lintons": {
        "path":      "data/linton.csv",
        "name_col":  "product_name",
        "price_col": "current_price",
        "brand_col": None,
    },
    "mydawa": {
        "path":      "data/mydawa_product_list.csv",
        "name_col":  "name",
        "price_col": "price_kes",
        "brand_col": "brand_name",
    },
}

OUTPUT_PATH = "data/competitor_prices.csv"


# ── Text helpers ───────────────────────────────────────────────────────────────
def _normalise(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _extract_size(s: str):
    """
    Extract (value, unit) from a product name.
    Returns None if no size found.
    Units: ml, g, gm, mg, kg, oz, l — single-letter units require word boundary.
    """
    pattern = (
        r"(\d+(?:\.\d+)?)\s*"
        r"(ml|gm|mg|kg|oz|pcs|tabs|caps|sachets|g(?!\w)|l(?!\w)|s(?!\w))"
    )
    matches = re.findall(pattern, s.lower())
    if not matches:
        return None
    # Prefer volume/weight units over pack counts
    for val, unit in matches:
        if unit in ("ml", "g", "gm", "mg", "kg", "l", "oz"):
            return float(val), unit
    return float(matches[0][0]), matches[0][1]


def _size_compatible(name_a: str, name_b: str) -> bool:
    """
    True if sizes are compatible.
    Rejects only when both names have a measurable size in the same unit
    and they differ by more than SIZE_TOLERANCE.
    """
    sa = _extract_size(name_a)
    sb = _extract_size(name_b)
    if sa is None or sb is None:
        return True
    val_a, unit_a = sa
    val_b, unit_b = sb
    if unit_a != unit_b:
        return True
    return abs(val_a - val_b) / max(val_a, val_b) <= SIZE_TOLERANCE


# ── Competitor loader ──────────────────────────────────────────────────────────
def _load_competitor(key: str) -> tuple:
    """
    Load and normalise one competitor CSV.
    Returns (names_norm, names_raw, prices, brand_names_norm | None)
    """
    cfg = SOURCES[key]
    if not os.path.exists(cfg["path"]):
        logging.warning(f"Competitor file not found: {cfg['path']}")
        return [], [], [], None

    df = pl.read_csv(cfg["path"], infer_schema_length=500)

    names_raw  = df[cfg["name_col"]].fill_null("").to_list()
    prices     = df[cfg["price_col"]].to_list()
    names_norm = [_normalise(n) for n in names_raw]

    brand_norm = None
    if cfg["brand_col"] and cfg["brand_col"] in df.columns:
        brands = df[cfg["brand_col"]].fill_null("").to_list()
        brand_norm = [
            _normalise(f"{b} {n}") for b, n in zip(brands, names_raw)
        ]

    return names_norm, names_raw, prices, brand_norm


# ── Per-SKU matcher ────────────────────────────────────────────────────────────
def _best_match(
    query_norm: str,
    query_raw:  str,
    names_norm: list,
    names_raw:  list,
    prices:     list,
    brand_norm: list = None,
) -> tuple:
    """
    Find best competitor price for one SKU.
    Returns (price | None, score)
    """
    best_score = 0
    best_price = None

    # Primary: match on product_name
    result = process.extractOne(
        query_norm, names_norm, scorer=fuzz.token_sort_ratio
    )
    if result and result[1] >= THRESHOLD:
        idx = names_norm.index(result[0])
        if _size_compatible(query_raw, names_raw[idx]):
            best_score = result[1]
            best_price = prices[idx]

    # Secondary: match on brand + name (MyDawa only)
    if brand_norm:
        alt_result = process.extractOne(
            query_norm, brand_norm, scorer=fuzz.token_sort_ratio
        )
        if alt_result and alt_result[1] > best_score and alt_result[1] >= THRESHOLD:
            idx = brand_norm.index(alt_result[0])
            if _size_compatible(query_raw, names_raw[idx]):
                best_score = alt_result[1]
                best_price = prices[idx]
                logging.debug(
                    f"Fallback (brand+name): '{query_raw}' "
                    f"-> '{names_raw[idx]}' score={best_score:.0f}"
                )

    return best_price, best_score


# ── Main entry point ───────────────────────────────────────────────────────────
def build_competitor_prices(
    targets_path: str = "data/analysis_targets_20.csv",
    output_path:  str = OUTPUT_PATH,
) -> pl.DataFrame:
    """
    Match PharmaPlus SKUs against all three competitor catalogs.
    Returns and saves the price comparison DataFrame.

    Args:
        targets_path : path to analysis_targets_20.csv (20 at-risk SKUs)
        output_path  : where to save competitor_prices.csv

    Returns:
        pl.DataFrame with columns:
            product_id, product_name, pharmaplus_price,
            goodlife, lintons, mydawa,
            market_low, price_gap_pct, price_position
    """
    # Load our 20 SKUs
    targets = (
        pl.read_csv(targets_path)
        .select(["product_id", "product_name", "selling_price"])
        .unique(subset=["product_id"])
    )

    # Load all three competitor catalogs
    comp_data = {key: _load_competitor(key) for key in SOURCES}

    results = []
    for row in targets.iter_rows(named=True):
        pid        = str(row["product_id"])
        raw_name   = row["product_name"]
        our_price  = float(row["selling_price"])
        query_norm = _normalise(raw_name)

        prices  = {}
        scores  = {}
        for key, (names_norm, names_raw, price_list, brand_norm) in comp_data.items():
            if not names_norm:
                prices[key] = None
                scores[key] = 0
                continue
            p, s = _best_match(
                query_norm, raw_name,
                names_norm, names_raw, price_list,
                brand_norm
            )
            prices[key] = p
            scores[key] = s

        # Derived fields
        comp_values   = [v for v in prices.values() if v is not None]
        market_low    = min(comp_values) if comp_values else None
        match_count   = len(comp_values)
        best_score    = max(scores.values()) if scores else 0

        if market_low is not None:
            gap_pct = (our_price - market_low) / market_low
            if gap_pct > 0.05:
                position = "ABOVE"
            elif gap_pct < -0.05:
                position = "BELOW"
            else:
                position = "MATCH"
        else:
            gap_pct  = None
            position = None

        results.append({
            "product_id":             pid,
            "product_name":           raw_name,
            "pharmaplus_price":       our_price,
            "goodlife":               prices.get("goodlife"),
            "lintons":                prices.get("lintons"),
            "mydawa":                 prices.get("mydawa"),
            "market_low":             market_low,
            "price_gap_pct":          round(gap_pct, 4) if gap_pct is not None else None,
            "price_position":         position,
            "match_score":            round(best_score, 1),
            "competitor_match_count": match_count,
        })

    df_out = pl.DataFrame(results)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_out.write_csv(output_path)

    matched = df_out.filter(pl.col("market_low").is_not_null()).shape[0]
    logging.info(
        f"Competitor pricing: {matched}/{df_out.shape[0]} SKUs matched. "
        f"Saved to {output_path}"
    )
    return df_out
