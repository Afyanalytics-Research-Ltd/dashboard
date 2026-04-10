"""
engine/catalogue_matcher.py

Improved matching strategy for low-overlap datasets (Tendri vs Pharmaplus):

The core problem: Tendri names like "ENSURE NUTRI VANNILLA 400GM" don't fuzzy-
match well against Pharmaplus names because:
  1. Brand names are often the most distinctive token but get diluted by dosage
  2. Pack size is in the name as text ("400GM") not a separate field
  3. Category-level matching is more reliable than name-level matching at scale

New strategy (in priority order):
  1. Exact normalised match (unchanged)
  2. Brand-anchored match: extract the first token as brand, find all Pharmaplus
     SKUs sharing that brand token, then fuzzy match within that subset only
  3. Full-name fuzzy match with a higher threshold (80) as fallback
  4. Category-only assignment: if name match fails but we can infer category
     from keywords, assign internal_category without a SKU match

This gives us useful category enrichment even when SKU-level matching fails,
which is the more important output for dead stock tier assignment.
"""

import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import re

MATCH_THRESHOLD  = 80
BRAND_THRESHOLD  = 72   # lower threshold when brand scope is already narrowed
UNMATCHED_MARKER = "__NO_MATCH__"

PHARMAPLUS_CATEGORY_MAP = {
    "prescription":           "Pharma",
    "otc":                    "Pharma",
    "antibiotic":             "Pharma",
    "analgesic":              "Pharma",
    "antimalaria":            "Pharma",
    "vitamin":                "Pharma",
    "chronic":                "Pharma",
    "medical device":         "Pharma",
    "eye care":               "Pharma",
    "ear care":               "Pharma",
    "dental":                 "Pharma",
    "wound":                  "Pharma",
    "first aid":              "Pharma",
    "skincare":               "Beauty & Cosmetics",
    "skin care":              "Beauty & Cosmetics",
    "suncare":                "Beauty & Cosmetics",
    "sun care":               "Beauty & Cosmetics",
    "hair care":              "Beauty & Cosmetics",
    "haircare":               "Beauty & Cosmetics",
    "makeup":                 "Beauty & Cosmetics",
    "cosmetic":               "Beauty & Cosmetics",
    "fragrance":              "Beauty & Cosmetics",
    "body care":              "Beauty & Cosmetics",
    "personal care":          "Beauty & Cosmetics",
    "feminine":               "Beauty & Cosmetics",
    "supplement":             "Vitamins & Supplements",
    "multivitamin":           "Vitamins & Supplements",
    "omega":                  "Vitamins & Supplements",
    "probiotic":              "Vitamins & Supplements",
    "mineral":                "Vitamins & Supplements",
    "bodybuilding":           "Body Building",
    "body building":          "Body Building",
    "protein":                "Body Building",
    "sports nutrition":       "Body Building",
    "fitness":                "Body Building",
    "pre-workout":            "Body Building",
    "baby":                   "Non-Pharma",
    "home care":              "Non-Pharma",
    "surgical":               "Non-Pharma",
    "diagnostic":             "Non-Pharma",
    "medical supply":         "Non-Pharma",
    "medical supplies":       "Non-Pharma",
}

# Keywords in product NAMES that strongly imply a category
# Used for category-only assignment when SKU matching fails
NAME_CATEGORY_KEYWORDS = {
    "Pharma": [
        "tablet", "tabs", "capsule", "caps", "syrup", "suspension", "injection",
        "amoxicillin", "augmentin", "metformin", "atorvastatin", "paracetamol",
        "ibuprofen", "omeprazole", "amlodipine", "metronidazole", "ciprofloxacin",
        "azithromycin", "lisinopril", "losartan", "salbutamol", "prednisolone",
        "diclofenac", "fluconazole", "albendazole", "artemether", "coartem",
        "zinc", "folic", "ferrous", "insulin", "vaccine", "serum", "inhaler",
        "suppository", "pessary", "lozenge", "drops", "ointment", "cream mg",
        "ml solution", "vial", "ampoule", "sachet oral",
    ],
    "Beauty & Cosmetics": [
        "lotion", "moisturiser", "moisturizer", "sunscreen", "spf", "serum face",
        "toner", "cleanser", "foundation", "lipstick", "mascara", "shampoo",
        "conditioner", "hair oil", "body wash", "shower gel", "deodorant",
        "antiperspirant", "perfume", "cologne", "face wash", "scrub", "mask",
        "neutrogena", "cerave", "nivea", "vaseline", "dove", "olay", "garnier",
        "loreal", "l'oreal", "pantene", "head shoulders",
    ],
    "Vitamins & Supplements": [
        "multivitamin", "omega 3", "omega-3", "vitamin c", "vitamin d",
        "vitamin b", "calcium", "magnesium", "probiotic", "collagen",
        "glucosamine", "fish oil", "evening primrose", "coq10", "lutein",
        "ensure", "pediasure", "boost", "fortisip", "complan",
        "pregnacare", "wellwoman", "wellman", "seven seas",
    ],
    "Body Building": [
        "whey", "protein powder", "creatine", "bcaa", "pre workout",
        "mass gainer", "fat burner", "l-carnitine", "glutamine",
        "testosterone booster", "gym", "sports", "isolate",
    ],
    "Non-Pharma": [
        "plaster of paris", "bandage", "gauze", "glove", "syringe",
        "needle", "catheter", "cannula", "stethoscope", "thermometer",
        "glucometer", "test strip", "diaper", "nappy", "baby formula",
        "sanitary", "cotton wool",
    ],
}

DEAD_STOCK_TIERS_BY_CATEGORY = {
    "Pharma":                 {"WATCH": 30, "ALERT": 60, "DEAD": 90},
    "Beauty & Cosmetics":     {"WATCH": 20, "ALERT": 40, "DEAD": 60},
    "Vitamins & Supplements": {"WATCH": 25, "ALERT": 50, "DEAD": 75},
    "Body Building":          {"WATCH": 20, "ALERT": 40, "DEAD": 60},
    "Non-Pharma":             {"WATCH": 15, "ALERT": 30, "DEAD": 45},
    "default":                {"WATCH": 30, "ALERT": 60, "DEAD": 90},
}


def _normalise(text: str) -> str:
    if pd.isna(text):
        return ""
    t = str(text).lower()
    t = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|ml|g|kg|iu|tabs?|caps?|pcs?|pk|gm)\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _extract_brand_token(norm_name: str) -> str:
    """First meaningful token — usually the brand name."""
    tokens = norm_name.split()
    return tokens[0] if tokens else ""


def _infer_category_from_name(product_name: str) -> str:
    """
    Infers internal_category from product name keywords alone.
    Used when SKU matching fails — still gives us category-aware thresholds.
    Returns 'Pharma' as default if no keyword matches.
    """
    name_lower = str(product_name).lower()
    for category, keywords in NAME_CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return category
    return "Pharma"


def _map_pharmaplus_category(raw_category: str) -> str:
    if pd.isna(raw_category):
        return "Pharma"
    cat = str(raw_category).lower().strip()
    for keyword, mapped in PHARMAPLUS_CATEGORY_MAP.items():
        if keyword in cat:
            return mapped
    return "Pharma"


def load_pharmaplus_catalogue(pharmaplus_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(pharmaplus_csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    df = df.rename(columns={
        "sku":                   "pharmaplus_sku",
        "price_kes":             "price_kes",
        "category":              "raw_category",
        "brand":                 "brand",
        "units_sold":            "units_sold",
        "is_on_promotion":       "is_on_promotion",
        "promo_discounted_value":"promo_discounted_value",
    })

    df["internal_category"] = df["raw_category"].apply(_map_pharmaplus_category)
    df["brand"]     = df.get("brand", pd.Series(dtype=str)).fillna("").str.strip().str.lower()
    df["norm_name"] = df["name"].apply(_normalise)
    df["brand_token"] = df["norm_name"].apply(_extract_brand_token)

    keep = [
        "pharmaplus_sku", "name", "brand", "brand_token", "internal_category",
        "raw_category", "price_kes", "is_on_promotion",
        "promo_discounted_value", "units_sold", "norm_name",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].drop_duplicates(subset=["pharmaplus_sku"]).reset_index(drop=True)


# def match_tendri_to_pharmaplus(
#     tendri_products: pd.DataFrame,
#     pharmaplus_catalogue: pd.DataFrame,
#     threshold: int = MATCH_THRESHOLD,
# ) -> pd.DataFrame:
#     """
#     Three-stage matching with category-only fallback.

#     Stage 1: Exact normalised name match
#     Stage 2: Brand-anchored fuzzy match (search within brand subset)
#     Stage 3: Full-catalogue fuzzy match at higher threshold
#     Stage 4 (fallback): No SKU match, but infer category from name keywords
#     """
#     tendri_products           = tendri_products.copy()
#     tendri_products["norm_name"]    = tendri_products["product_name"].apply(_normalise)
#     tendri_products["brand_token"]  = tendri_products["norm_name"].apply(_extract_brand_token)

#     # Build lookup structures
#     exact_lookup   = pharmaplus_catalogue.set_index("norm_name")
#     all_pp_names   = pharmaplus_catalogue["norm_name"].tolist()

#     # Brand token → subset of catalogue rows
#     brand_groups = {}
#     for bt, grp in pharmaplus_catalogue.groupby("brand_token"):
#         brand_groups[bt] = grp

#     rows = []
#     for _, row in tendri_products.iterrows():
#         pid   = row["product_id"]
#         pname = row["product_name"]
#         norm  = row["norm_name"]
#         bt    = row["brand_token"]

#         if not norm:
#             rows.append(_category_only_row(pid, pname))
#             continue

#         # Stage 1: exact
#         if norm in exact_lookup.index:
#             cat_row = exact_lookup.loc[norm]
#             if isinstance(cat_row, pd.DataFrame):
#                 cat_row = cat_row.iloc[0]
#             rows.append(_match_row(pid, pname, cat_row, 100, "exact"))
#             continue

#         # Stage 2: brand-anchored fuzzy
#         matched = None
#         if bt and bt in brand_groups:
#             subset      = brand_groups[bt]
#             subset_names = subset["norm_name"].tolist()
#             result = process.extractOne(norm, subset_names, scorer=fuzz.token_sort_ratio)
#             if result and result[1] >= BRAND_THRESHOLD:
#                 best_name = result[0]
#                 cat_row   = subset[subset["norm_name"] == best_name].iloc[0]
#                 matched   = _match_row(pid, pname, cat_row, result[1], "fuzzy_brand")

#         # Stage 3: full-catalogue fuzzy (only if brand match failed)
#         if matched is None:
#             result = process.extractOne(norm, all_pp_names, scorer=fuzz.token_sort_ratio)
#             if result and result[1] >= threshold:
#                 best_name = result[0]
#                 if best_name in exact_lookup.index:
#                     cat_row = exact_lookup.loc[best_name]
#                     if isinstance(cat_row, pd.DataFrame):
#                         cat_row = cat_row.iloc[0]
#                     matched = _match_row(pid, pname, cat_row, result[1], "fuzzy")

#         if matched:
#             rows.append(matched)
#         else:
#             # Stage 4: category-only from name keywords — no SKU, but still useful
#             rows.append(_category_only_row(pid, pname))

#     return pd.DataFrame(rows)


def _match_row(pid, pname, cat_row, score, match_type) -> dict:
    return {
        "product_id":            pid,
        "product_name":          pname,
        "pharmaplus_sku":        cat_row.get("pharmaplus_sku", UNMATCHED_MARKER),
        "pharmaplus_name":       cat_row.get("name", ""),
        "pharmaplus_brand":      cat_row.get("brand", ""),
        "internal_category":     cat_row.get("internal_category", "Pharma"),
        "match_score":           score,
        "match_type":            match_type,
        "price_kes_pharmaplus":  cat_row.get("price_kes", np.nan),
        "units_sold_pharmaplus": cat_row.get("units_sold", np.nan),
    }


def _category_only_row(pid, pname) -> dict:
    """No SKU match but infer category from name — still useful for tier assignment."""
    inferred_cat = _infer_category_from_name(pname)
    return {
        "product_id":            pid,
        "product_name":          pname,
        "pharmaplus_sku":        UNMATCHED_MARKER,
        "pharmaplus_name":       "",
        "pharmaplus_brand":      "",
        "internal_category":     inferred_cat,
        "match_score":           0,
        "match_type":            "category_inferred",
        "price_kes_pharmaplus":  np.nan,
        "units_sold_pharmaplus": np.nan,
    }


def basket_gap_report(
    pharmaplus_catalogue: pd.DataFrame,
    match_table: pd.DataFrame,
) -> pd.DataFrame:
    matched_skus = set(
        match_table[match_table["pharmaplus_sku"] != UNMATCHED_MARKER]["pharmaplus_sku"]
    )
    gaps = pharmaplus_catalogue[~pharmaplus_catalogue["pharmaplus_sku"].isin(matched_skus)]
    return (
        gaps.groupby("internal_category")
        .agg(sku_count=("pharmaplus_sku", "count"))
        .sort_values("sku_count", ascending=False)
        .reset_index()
    )


def get_category_tiers(internal_category: str) -> dict:
    return DEAD_STOCK_TIERS_BY_CATEGORY.get(
        internal_category,
        DEAD_STOCK_TIERS_BY_CATEGORY["default"],
    )