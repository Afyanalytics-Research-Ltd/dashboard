"""
engine/price_signal.py

Three-layer price signal with full coverage — every SKU gets a signal.

Layer 1 — Brand-scoped fuzzy match
    Narrow candidate pool to same brand + category before fuzzy scoring.
    Threshold lowered to 65 within scoped pool (safe because brand already matches).

Layer 2 — Key token match
    Extract brand token + pack size token. Match on these independently.
    Catches cases where full name diverges but key identifiers match.

Layer 3 — Category benchmark
    For every unmatched SKU, compare price against the median competitor
    price in its category. Flagged as "above/below category median".
    match_method = "category_benchmark" so UI can show confidence level.

Source routing by category:
    Pharma                → Mydawa  (primary)
    Beauty & Cosmetics    → Goodlife (primary)
    Vitamins & Supplements→ Goodlife + Linton
    Body Building         → Linton  (primary)
    Non-Pharma            → Goodlife
"""

import os
import re
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz


# Categories where category-level benchmarking is unreliable
# These have too much price dispersion within the category to benchmark meaningfully
ABOVE_THRESHOLD = 8.0
BELOW_THRESHOLD = -8.0

SCOPED_MATCH_THRESHOLD = 65
FULL_MATCH_THRESHOLD   = 78   # full catalogue fallback

BENCHMARK_UNRELIABLE_KEYWORDS = [
    "perfume", "eau de", "cologne", "edt", "edp", "fragrance",
    "parfum", "deodorant spray",                          # luxury fragrance
    "electric toothbrush", "rechargeable toothbrush",     # premium dental devices
    "smartwatch", "blood pressure monitor", "nebuliser",  # medical devices
    "premium", "luxury", "gold standard",                 # premium supplements
]

def _is_benchmark_unreliable(product_name: str) -> bool:
    """Returns True if this SKU's category benchmark is too noisy to trust."""
    name = str(product_name).lower()
    return any(kw in name for kw in BENCHMARK_UNRELIABLE_KEYWORDS)



CATEGORY_SOURCE_PRIORITY = {
    "Pharma":                  ["mydawa"],
    "Beauty & Cosmetics":      ["goodlife"],
    "Vitamins & Supplements":  ["goodlife", "linton"],
    "Body Building":           ["linton", "goodlife"],
    "Non-Pharma":              ["goodlife"],
}


# ─── NORMALISATION ────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    if pd.isna(text):
        return ""
    t = str(text).lower()
    t = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|ml|g|kg|iu|tabs?|caps?|pcs?|pk|gm|l)\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _brand_token(name: str) -> str:
    """First token of normalised name — usually the brand."""
    tokens = _norm(name).split()
    return tokens[0] if tokens else ""


def _pack_token(name: str) -> str:
    """Extract pack size token e.g. '250ml', '1kg', '100g'."""
    m = re.search(r"\b(\d+(\.\d+)?\s*(ml|g|kg|l|mg|mcg|iu|tabs?|caps?))\b",
                  str(name).lower())
    return m.group(0).replace(" ", "") if m else ""


# ─── LOADERS ─────────────────────────────────────────────────────────────────

def _load_source(path: str, source_name: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Normalise product_name column — Mydawa uses 'name', others use 'product_name'
        if "product_name" not in df.columns and "name" in df.columns:
            df = df.rename(columns={"name": "product_name"})

        # Normalise price column name across sources
        for col in ["current_price", "price_kes", "price"]:
            if col in df.columns:
                df["price"] = pd.to_numeric(df[col], errors="coerce")
                break

        # Promo price
        for col in ["discount_price", "promo_price", "original_price"]:
            if col in df.columns:
                df["promo_price"] = pd.to_numeric(df[col], errors="coerce")
                break
        if "promo_price" not in df.columns:
            df["promo_price"] = np.nan

        # Has promo
        if "discount_badge" in df.columns:
            df["has_promo"] = df["discount_badge"].notna() & (df["discount_badge"].astype(str).str.strip() != "")
        elif "is_on_promotion" in df.columns:
            df["has_promo"] = df["is_on_promotion"].fillna(False).astype(bool)
        else:
            df["has_promo"] = False

        # Effective price = promo if available and lower
        df["effective_price"] = np.where(
            df["promo_price"].notna() & (df["promo_price"] < df["price"]),
            df["promo_price"],
            df["price"],
        )

        df["source"]      = source_name
        df["norm_name"]   = df["product_name"].apply(_norm)
        df["brand_tok"]   = df["product_name"].apply(_brand_token)
        df["pack_tok"]    = df["product_name"].apply(_pack_token)

        return df[["product_name", "norm_name", "brand_tok", "pack_tok",
                   "price", "effective_price", "has_promo", "source"]].dropna(subset=["effective_price"])
    except Exception as e:
        print(f"[price_signal] Warning: could not load {source_name}: {e}")
        return pd.DataFrame()


def load_competitor_data(
    mydawa_path:   str = None,
    goodlife_path: str = None,
    linton_path:   str = None,
) -> dict[str, pd.DataFrame]:
    return {
        k: v for k, v in {
            "mydawa":   _load_source(mydawa_path,   "mydawa"),
            "goodlife": _load_source(goodlife_path, "goodlife"),
            "linton":   _load_source(linton_path,   "linton"),
        }.items() if not v.empty
    }


# ─── CATEGORY BENCHMARKS ─────────────────────────────────────────────────────

def _build_category_benchmarks(
    competitor_sources: dict[str, pd.DataFrame],
) -> dict[str, dict]:
    """
    Builds median price per (category, source) using ORIGINAL (non-promo) prices.
    Using effective/discounted prices as the benchmark artificially inflates
    the "above market" signal — a product at full price vs a competitor's
    sale price is not meaningfully overpriced.
    """
    from engine.catalogue_matcher import _infer_category_from_name

    benchmarks = {}
    for src_name, df in competitor_sources.items():
        if df.empty:
            continue
        df = df.copy()
        # Use original price for benchmark, not promo price
        benchmark_price = df["price"].where(df["price"].notna(), df["effective_price"])
        df["benchmark_price"] = benchmark_price
        df["inferred_cat"] = df["product_name"].apply(_infer_category_from_name)
        for cat, grp in df.groupby("inferred_cat"):
            prices = grp["benchmark_price"].dropna()
            prices = prices[prices > 0]
            if len(prices) < 5:
                continue
            # Use 75th percentile as benchmark, not median
            # This avoids penalising PharmaPlus for competitor loss-leaders
            key = (cat, src_name)
            benchmarks[key] = {
                "median": prices.quantile(0.75),
                "p25":    prices.quantile(0.50),
                "p75":    prices.quantile(0.90),
                "n":      len(prices),
            }
    return benchmarks


# ─── MATCHING ────────────────────────────────────────────────────────────────

def _layer1_brand_scoped(
    norm_name: str,
    brand_tok: str,
    source_df: pd.DataFrame,
) -> pd.Series | None:
    """Brand-scoped fuzzy match — narrow to same brand first."""
    if not brand_tok or source_df.empty:
        return None
    scoped = source_df[source_df["brand_tok"] == brand_tok]
    if scoped.empty:
        # Try partial brand match (e.g. "neutrogena" in "neutrogena ultra")
        scoped = source_df[source_df["brand_tok"].str.startswith(brand_tok[:4])]
    if scoped.empty:
        return None
    names  = scoped["norm_name"].tolist()
    result = process.extractOne(norm_name, names, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= SCOPED_MATCH_THRESHOLD:
        row = scoped[scoped["norm_name"] == result[0]]
        if not row.empty:
            return row.loc[row["effective_price"].idxmin()]
    return None


def _layer2_token_match(
    brand_tok: str,
    pack_tok:  str,
    source_df: pd.DataFrame,
) -> pd.Series | None:
    """Match on brand token + pack size token."""
    if not brand_tok or source_df.empty:
        return None
    mask = source_df["brand_tok"] == brand_tok
    if pack_tok:
        mask = mask & source_df["pack_tok"].str.contains(pack_tok[:4], na=False)
    candidates = source_df[mask]
    if candidates.empty:
        return None
    return candidates.loc[candidates["effective_price"].idxmin()]


def _layer3_benchmark(
    pharmaplus_price: float,
    category:         str,
    source_order:     list[str],
    benchmarks:       dict,
) -> dict | None:
    """Category-level benchmark comparison."""
    if pd.isna(pharmaplus_price):
        return None
    for src in source_order:
        key = (category, src)
        if key in benchmarks:
            b = benchmarks[key]
            median = b["median"]
            if median > 0:
                pct = ((pharmaplus_price - median) / median) * 100
                return {
                    "market_price":    round(median, 2),
                    "pct":             round(pct, 1),
                    "source":          src,
                    "n_products":      b["n"],
                    "match_method":    "category_benchmark",
                }
    return None


# ─── MAIN BUILDER ────────────────────────────────────────────────────────────

def build_price_signal(
    match_table:        pd.DataFrame,
    competitor_sources: dict[str, pd.DataFrame],
    dead_stock:         pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Builds price signal for every SKU in match_table.
    Every row gets a signal — either SKU match, token match, or category benchmark.
    """
    if match_table.empty or not competitor_sources:
        return pd.DataFrame()

    benchmarks = _build_category_benchmarks(competitor_sources)
    rows       = []

    for _, sku in match_table.iterrows():
        pid       = sku["product_id"]
        pname     = str(sku.get("product_name", ""))
        category  = str(sku.get("internal_category", "Pharma"))
        pp_price  = float(sku.get("price_kes_pharmaplus") or 0) or np.nan
        brand_tok = _brand_token(pname)
        pack_tok  = _pack_token(pname)
        norm      = _norm(pname)

        source_order = CATEGORY_SOURCE_PRIORITY.get(category, ["goodlife"])

        best_price   = np.nan
        best_source  = None
        match_method = "no_match"
        has_promo    = False

        # Layer 1 + 2 across priority sources
        for src_name in source_order:
            if src_name not in competitor_sources:
                continue
            src_df = competitor_sources[src_name]

            match = _layer1_brand_scoped(norm, brand_tok, src_df)
            if match is None:
                match = _layer2_token_match(brand_tok, pack_tok, src_df)

            if match is not None:
                mp = match["effective_price"]
                if pd.isna(best_price) or mp < best_price:
                    best_price   = mp
                    best_source  = src_name
                    match_method = "sku_match"
                    has_promo    = bool(match.get("has_promo", False))
                break  # found a match in priority source

        # Layer 3 — category benchmark fallback
        # Suppress for SKUs where category benchmark is known to be unreliable
        if pd.isna(best_price) and not _is_benchmark_unreliable(pname):
            bench = _layer3_benchmark(pp_price, category, source_order, benchmarks)
            if bench:
                best_price   = bench["market_price"]
                best_source  = bench["source"]
                match_method = "category_benchmark"

        # Compute signal
        if not pd.isna(pp_price) and not pd.isna(best_price) and best_price > 0:
            pct = ((pp_price - best_price) / best_price) * 100
        else:
            pct = np.nan

        if pd.isna(pct):
            signal = "unknown"
        elif pct > ABOVE_THRESHOLD:
            signal = "above"
        elif pct < BELOW_THRESHOLD:
            signal = "below"
        else:
            signal = "at"

        rows.append({
            "product_id":               pid,
            "product_name":             pname,
            "internal_category":        category,
            "pharmaplus_price_kes":     pp_price,
            "market_price_kes":         round(best_price, 2) if not pd.isna(best_price) else np.nan,
            "primary_competitor":       best_source,
            "price_vs_market_pct":      round(pct, 1) if not pd.isna(pct) else np.nan,
            "price_signal":             signal,
            "match_method":             match_method,
            "competitor_promo_active":  has_promo,
        })

    signal_df = pd.DataFrame(rows)

    if dead_stock is not None and not signal_df.empty:
        signal_df = _add_freeze_hypothesis(signal_df, dead_stock)

    return signal_df


def _add_freeze_hypothesis(signal_df: pd.DataFrame, dead_stock: pd.DataFrame) -> pd.DataFrame:
    ds = dead_stock[["product_id", "tier"]].drop_duplicates("product_id")
    merged = signal_df.merge(ds, on="product_id", how="left")

    def hypothesis(row):
        sig   = row.get("price_signal", "unknown")
        tier  = row.get("tier", "")
        promo = row.get("competitor_promo_active", False)
        pct   = row.get("price_vs_market_pct", 0) or 0
        meth  = row.get("match_method", "")

        # Category benchmarks are estimates — only flag price-driven at higher confidence
        price_threshold = 25.0 if meth == "category_benchmark" else 15.0

        if sig == "unknown":
            return "unknown"
        if sig == "above" and abs(pct) >= price_threshold and tier == "DEAD":
            return "price"
        if sig == "above" and abs(pct) >= price_threshold and promo:
            return "both"
        if sig == "above" and abs(pct) >= price_threshold:
            return "price"
        if sig in ("at", "below") and tier in ("DEAD", "ALERT"):
            return "demand"
        return "unknown"

    merged["freeze_hypothesis"] = merged.apply(hypothesis, axis=1)
    return merged.drop(columns=["tier"], errors="ignore")


def price_signal_summary(signal_df: pd.DataFrame) -> dict:
    if signal_df.empty:
        return {"above_market": 0, "at_market": 0, "below_market": 0,
                "unknown": 0, "with_competitor_promos": 0, "category_benchmarks": 0}
    counts = signal_df["price_signal"].value_counts().to_dict()
    return {
        "above_market":        counts.get("above", 0),
        "at_market":           counts.get("at", 0),
        "below_market":        counts.get("below", 0),
        "unknown":             counts.get("unknown", 0),
        "with_competitor_promos": int(signal_df.get("competitor_promo_active", pd.Series(False)).sum()),
        "category_benchmarks": int((signal_df.get("match_method", pd.Series()) == "category_benchmark").sum()),
    }