"""
Dispensing + Inventory Simulation
===================================
PURPOSE:
  Generate simulated dispensing and inventory data for Beauty Products,
  Vitamins & Supplements, and Body Building categories — which don't exist
  in the real hospital dispensing data but would exist in a Pharmaplus-style
  retail pharmacy in Embu.

  The simulation is designed to tell a specific business story:
    1. Fast movers vs slow movers vs dead stock
    2. Slow movers split by cause: overstocking vs no demand
    3. Profitability intelligence: sales != profit (margin contribution by product)
    4. Category mix: beauty/supplements have better margins than pharma
    5. Generic vs branded substitution opportunities with KES savings

ANCHOR:
  Pharma = 48% = 296,849 units/month (from real data, facility 4 + 5 combined)
  All new categories sized proportionally.

FACILITY SPLIT (from real patient data):
  Facility 4: ~75% of volume (976 unique patients)
  Facility 5: ~25% of volume (323 unique patients)

OUTPUTS (all saved to tendri/data/):
  1. simulated_dispensing_aggregate_DATE.csv    -> monthly_dispensing_aggregate schema
  2. simulated_fact_dispensing_DATE.csv         -> fact_dispensing schema (no store_id)
  3. simulated_inventory_snapshot_DATE.csv      -> fact_inventory_snapshot schema (no store_id)
  4. simulated_product_intelligence_DATE.csv    -> profitability + generic/branded flags

SCHEMA:
  simulated_dispensing_aggregate matches monthly_dispensing_aggregate exactly:
    facility_id, months, product_id, new_category_name, parent_category_name,
    correct_therapeutic_class, total_qty_dispensed, unique_patients, avg_qty_per_patient

  No store_id anywhere — removed by design.

Requirements:
  pip install pandas numpy
"""

import os
import glob
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, date

np.random.seed(42)
TODAY   = datetime.now().strftime("%Y-%m-%d")
OUT_DIR = "tendri\\data"
os.makedirs(OUT_DIR, exist_ok=True)

# ── File paths ────────────────────────────────────────────────────────────────
PHARMAPLUS_PATH = "tendri\\data\\pharmaplus_product_list.csv"
GOODLIFE_PATH   = "tendri\\data\\goodlife_price_list.csv"
MYDAWA_PATH     = "tendri\\data\\mydawa_product_list.csv"
LINTON_PATH     = "tendri\\data\\linton.csv"
EMBU_GT_PATH    = "embu_trends/embu_trends_category_index_*.csv"

# ── Facility IDs (no store_id) ────────────────────────────────────────────────
FACILITY_IDS          = [4, 5]
FACILITY_VOLUME_SPLIT = {4: 0.75, 5: 0.25}   # from real patient counts
FACILITY_PATIENT_BASE = {4: 8000, 5: 3000}

# ── Simulation period (12+ months for KNN steady-state window) ────────────────
SIM_START = "2023-01-01"
SIM_END   = datetime.now().strftime("%Y-%m-01")

# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY MIX — anchored to real pharma volume
# ══════════════════════════════════════════════════════════════════════════════
PHARMA_MONTHLY_VOLUME = 30_328  # median combined volume, hospital-only filtered

CATEGORY_MIX = {
    "Beauty Products":        0.18,  # 18% of pharma
    "Vitamins & Supplements": 0.22,  # 22% of pharma
    "Body Building":          0.04,  # 4% of pharma
}
CATEGORY_TARGETS = {
    cat: round(PHARMA_MONTHLY_VOLUME * ratio)
    for cat, ratio in CATEGORY_MIX.items()
}

# ══════════════════════════════════════════════════════════════════════════════
# INVENTORY HEALTH PROFILES
# Deliberately structured to demonstrate the business intelligence story:
#   Overstocked slow: stock = 6x velocity -> purchasing problem
#   No-demand slow:   stock = 1.5x velocity -> market fit problem
# These look identical in dispensing data but are distinguishable only via
# the inventory snapshot — which is the core insight the dashboard surfaces.
# ══════════════════════════════════════════════════════════════════════════════
HEALTH_PROFILES = {
    "fast_mover":       {"pct": 0.20, "vol_share": 0.80, "stock_ratio": 1.2},
    "healthy_slow":     {"pct": 0.25, "vol_share": 0.12, "stock_ratio": 1.5},
    "overstocked_slow": {"pct": 0.20, "vol_share": 0.05, "stock_ratio": 6.0},
    "no_demand_slow":   {"pct": 0.20, "vol_share": 0.02, "stock_ratio": 1.5},
    "dead_stock":       {"pct": 0.15, "vol_share": 0.01, "stock_ratio": 8.0},
}

# ══════════════════════════════════════════════════════════════════════════════
# REPURCHASE FREQUENCY
# Drives unique_patients derivation.
# Monthly repurchase: same customers return -> LOW unique_patients relative to qty
# Biannual repurchase: mostly new customers each month -> HIGH unique_patients
# ══════════════════════════════════════════════════════════════════════════════
REPURCHASE_KEYWORDS = {
    "monthly": [
        "sunscreen","spf","sunblock","face wash","cleanser",
        "vitamin c","zinc","vitamin d","b complex","multivitamin",
        "prenatal","folic","ferrous","omega 3","fish oil","cod liver",
        "whey","protein powder","creatine","bcaa","pre workout","mass gainer",
        "probiotic","immune booster","shampoo","conditioner",
    ],
    "quarterly": [
        "moisturiser","moisturizer","body lotion","serum","toner","exfoliat",
        "collagen","biotin","calcium","magnesium",
        "hair mask","deep conditioner","hair growth",
        "foundation","bb cream","cc cream",
    ],
    "biannual": [
        "lipstick","lip gloss","mascara","eyeliner",
        "eyeshadow","palette","blush","highlighter",
        "concealer","setting powder","setting spray",
        "fragrance","perfume","cologne","nail polish",
        "hair relaxer","hair dye","hair colour",
    ],
}

# Avg units purchased per visit by repurchase cycle
AVG_UNITS_PER_VISIT = {
    "monthly":   (1.5, 2.5),
    "quarterly": (1.0, 2.0),
    "biannual":  (1.0, 1.3),
}

REPURCHASE_VELOCITY_MULT = {
    "monthly":   1.00,
    "quarterly": 0.50,
    "biannual":  0.20,
}

def get_repurchase_cycle(product_name):
    name = str(product_name).lower()
    for cycle, keywords in REPURCHASE_KEYWORDS.items():
        if any(kw in name for kw in keywords):
            return cycle
    return "quarterly"

def compute_unique_patients(qty, repurchase_cycle):
    """
    Derive unique_patients from qty and repurchase cycle.
    avg_qty_per_patient is ALWAYS derived (qty / unique_patients), never simulated.
    """
    min_upv, max_upv = AVG_UNITS_PER_VISIT.get(repurchase_cycle, (1.0, 2.0))
    avg_upv  = np.random.uniform(min_upv, max_upv)
    patients = max(int(qty / avg_upv), 1)
    return min(patients, qty)

# ══════════════════════════════════════════════════════════════════════════════
# PROFITABILITY — gross margin by price tier
# Story: budget fast movers have lower margin than mid-tier moderate sellers.
# Generics always earn more margin than branded equivalents.
# ══════════════════════════════════════════════════════════════════════════════
def gross_margin(price, is_generic):
    if price < 800:
        margin = np.random.uniform(0.18, 0.22)
    elif price < 3000:
        margin = np.random.uniform(0.28, 0.35)
    else:
        margin = np.random.uniform(0.38, 0.48)
    if is_generic:
        margin += np.random.uniform(0.08, 0.12)
    return round(min(margin, 0.65), 4)

def margin_tier(price):
    if price < 800:   return "budget"
    if price < 3000:  return "mid"
    return "premium"

# ══════════════════════════════════════════════════════════════════════════════
# GENERIC vs BRANDED DETECTION (all categories)
# ══════════════════════════════════════════════════════════════════════════════
GENERIC_TERMS = [
    "amoxicillin","ampicillin","azithromycin","ciprofloxacin","metronidazole",
    "doxycycline","cotrimoxazole","artemether","quinine","metformin",
    "glibenclamide","insulin","amlodipine","atenolol","losartan","lisinopril",
    "omeprazole","pantoprazole","ibuprofen","paracetamol","diclofenac",
    "salbutamol","prednisolone",
    "vitamin c","vitamin d","zinc sulfate","calcium carbonate","magnesium oxide",
    "omega 3","fish oil","multivitamin","biotin","folic acid","ferrous sulfate",
    "b complex","cod liver oil","collagen peptide",
    "glycerin","petroleum jelly","aloe vera gel","calamine lotion",
    "salicylic acid","benzoyl peroxide",
    "whey protein concentrate","creatine monohydrate","bcaa","glutamine powder",
]

KNOWN_BRANDS = [
    "augmentin","coartem","flagyl","ciproxin","zithromax","glucophage",
    "norvasc","cozaar","zestril","losec","voltaren",
    "redoxon","berocca","centrum","seven seas","supradyn","vitabiotics",
    "perfectil","nourkrin","neocell","neurobion","becosules","slow-mag",
    "blackmores","solgar","nature made","nordic naturals",
    "nivea","neutrogena","cerave","olay","vaseline","la roche-posay",
    "bioderma","eucerin","garnier","maybelline","revlon","loreal",
    "mac cosmetics","nyx","rimmel","cantu","dark and lovely","pantene",
    "dove","sunsilk","schwarzkopf","tresemme","the ordinary","cosrx",
    "optimum nutrition","usn","bsn","muscletech","evox","dymatize",
    "myprotein","serious mass","scivation","cellucor","ghost",
]

def detect_generic(product_name, brand_name=""):
    name  = str(product_name).lower()
    brand = str(brand_name).lower()
    if any(b in name or b in brand for b in KNOWN_BRANDS):
        return False
    if any(g in name for g in GENERIC_TERMS):
        return True
    return False

def generic_opportunity(row, same_class_products):
    """
    Conservative 60% switch rate assumption:
    Not all patients will accept a generic substitute.
    """
    if row["is_generic"]:
        return 0, None
    generics = same_class_products[
        same_class_products["is_generic"] &
        (same_class_products["product_id"] != row["product_id"])
    ]
    if generics.empty:
        return 0, None
    cheapest = generics.nsmallest(1, "price").iloc[0]
    saving   = (row["price"] - cheapest["price"]) * row["monthly_velocity"] * 0.60
    return max(round(saving, 2), 0), cheapest["product_name"]

# ══════════════════════════════════════════════════════════════════════════════
# SEASONAL PATTERNS (Kenya-specific)
# ══════════════════════════════════════════════════════════════════════════════
SEASONAL = {
    "Beauty Products":        {1:1.10,2:1.25,3:1.05,4:0.95,5:0.90,6:0.95,
                               7:1.05,8:1.05,9:1.10,10:1.05,11:1.10,12:1.30},
    "Vitamins & Supplements": {1:1.25,2:1.05,3:0.95,4:0.90,5:0.90,6:1.15,
                               7:1.20,8:1.15,9:1.00,10:1.00,11:1.00,12:1.05},
    "Body Building":          {1:1.45,2:1.25,3:1.05,4:0.90,5:0.85,6:0.90,
                               7:0.90,8:0.95,9:1.20,10:1.05,11:1.00,12:1.00},
}

KENYA_HOLIDAYS = [(1,1),(2,14),(4,1),(6,1),(10,20),(12,12),(12,25)]

def promo_mult(is_on_promo, month_ts):
    """
    Near holiday (within 21 days): demand-driven promo -> +15%
    Off-holiday: clearance signal (product not moving) -> -20%
    """
    if not is_on_promo:
        return 1.0
    for hm, hd in KENYA_HOLIDAYS:
        try:
            hdate = date(month_ts.year, hm, hd)
            if abs((month_ts.date() - hdate).days) <= 21:
                return 1.15
        except ValueError:
            continue
    return 0.80

# Vitamins habit formation: repeat purchase grows over months 1-3
HABIT = {1: 0.70, 2: 0.90, 3: 1.00}

# BB income sensitivity — two separate effects:
#
# 1. PAYDAY BOOST (every month, first week effect)
#    Most Kenyan salaries paid on 27th–31st of the month.
#    Customers spend on discretionary items (BB products) in the
#    week after payday — i.e. the first week of the following month.
#    Applied as a small uplift every month via the simulation loop.
BB_PAYDAY_BOOST = 1.10   # +10% every month (first-week-of-month effect)
#
# 2. BONUS MONTH BOOST (specific months only)
#    Larger uplift in months when bonuses or salary reviews are common in Kenya:
#      January  — end-of-year bonus spending carry-over + new year resolutions
#      April    — civil service and NGO annual reviews, Easter spending
#      August   — mid-year reviews, back-to-school adjacent spending
#      December — Christmas bonuses (captured in seasonal multiplier too)
BB_BONUS_MONTHS = {1: 1.20, 4: 1.15, 8: 1.12, 12: 1.18}
# Values represent additional multiplier on top of the payday boost.
# January gets the highest boost (new year gym rush + bonus carry-over).

# Store maturity ramp-up — 12 months
# Reflects the KNN model's steady-state window (months 4–12).
# Full maturity reached at month 12, not month 6.
#
# Beauty ramps slowest: exploratory category, customers need to try
# products, build trust, then return. Brand switching is high.
#
# Vitamins ramp fastest: once a customer finds a supplement that works,
# they're on a monthly repurchase cycle quickly. Habit forms in 1–3 months.
#
# Body Building is in between: loyal once committed, but the customer
# base is smaller and takes longer to discover a new pharmacy.
STORE_MATURITY = {
    "Beauty Products": {
        1:0.15, 2:0.25, 3:0.40, 4:0.55, 5:0.65, 6:0.74,
        7:0.82, 8:0.88, 9:0.93,10:0.96,11:0.98,12:1.00,
    },
    "Vitamins & Supplements": {
        1:0.30, 2:0.45, 3:0.58, 4:0.68, 5:0.76, 6:0.83,
        7:0.89, 8:0.93, 9:0.96,10:0.98,11:0.99,12:1.00,
    },
    "Body Building": {
        1:0.25, 2:0.38, 3:0.50, 4:0.62, 5:0.72, 6:0.80,
        7:0.86, 8:0.91, 9:0.95,10:0.97,11:0.99,12:1.00,
    },
}

# GT slope multipliers
GT_SLOPE_BOOST    = 1.25
GT_SLOPE_PENALTY  = 0.80

# Dead stock — category-aware thresholds
# 90 days is not universal. What counts as dead stock varies by category
# based on typical repurchase frequency and product shelf life:
#
#   Vitamins & Supplements: 90 days
#     Monthly repurchase products — 3 months with no movement is a clear signal
#
#   Beauty Products: 120 days
#     Slower turnover than vitamins, longer shelf life, but 4 months
#     with no movement means the product is genuinely not selling
#
#   Body Building: 150 days
#     Niche market, slower turns are expected. 5 months is the flag.
#     Some BB products (e.g. mass gainers) move slowly by nature.
#
# The simulation pushes last-sold dates back by these thresholds so the
# dashboard's dead stock detection logic fires correctly per category.
DEAD_STOCK_THRESHOLD_DAYS = {
    "Beauty Products":        120,
    "Vitamins & Supplements":  90,
    "Body Building":          150,
}

# ══════════════════════════════════════════════════════════════════════════════
# STATIC MULTIPLIER CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
BRAND_TIER_MULT = {"budget": 1.20, "mid": 1.00, "premium": 0.70}

PHARMACIST_UPLIFT = {
    "Vitamins & Supplements": 1.18,
    "Body Building":          1.08,
    "Beauty Products":        1.05,
}

SPILLOVER_KEYWORDS = [
    "vitamin c","zinc","probiotic","calcium","vitamin d",
    "folic","ferrous","omega","prenatal",
]

IMPULSE_MAX_PRICE = 500

BASKET_PAIRS = [
    # ── Beauty — skincare routine pairings ────────────────────────────────────
    ("sunscreen","moisturiser"),    ("sunscreen","moisturizer"),
    ("vitamin c serum","sunscreen"),# vitamin C AM + SPF always follows
    ("retinol","moisturiser"),      # retinol dries skin, moisturiser follows
    ("retinol","moisturizer"),
    ("face wash","toner"),          # cleanse-tone routine
    ("cleanser","toner"),
    ("toner","serum"),
    ("primer","foundation"),        # makeup base routine
    ("setting powder","foundation"),
    ("blush","highlighter"),        # colour + glow stack
    ("eye cream","concealer"),      # under-eye routine
    ("lip balm","lip scrub"),       # lip care routine
    # ── Beauty — haircare pairings ────────────────────────────────────────────
    ("shampoo","conditioner"),
    ("hair oil","shampoo"),
    ("hair mask","conditioner"),
    # ── Beauty — makeup pairings ──────────────────────────────────────────────
    ("foundation","concealer"),
    ("mascara","eyeliner"),
    # ── Vitamins — clinical absorption pairings ───────────────────────────────
    ("vitamin c","zinc"),           # immune stack
    ("vitamin d","calcium"),        # bone health stack
    ("vitamin d","magnesium"),      # vitamin D absorption needs magnesium
    ("iron","vitamin c"),           # vitamin C improves iron absorption
    ("ferrous","vitamin c"),        # same — ferrous = iron supplement
    ("omega 3","vitamin d"),        # often recommended together
    ("collagen","vitamin c"),       # vitamin C needed for collagen synthesis
    ("calcium","vitamin d"),
    ("zinc","vitamin c"),
    ("b complex","magnesium"),      # energy + nervous system stack
    ("prenatal","folic"),           # pregnancy supplementation
    ("probiotic","prebiotic"),      # gut health stack
    # ── Body Building — training stack pairings ───────────────────────────────
    ("whey","creatine"),            # most common BB stack
    ("protein","creatine"),
    ("pre workout","protein"),      # pre + post workout
    ("pre workout","bcaa"),
    ("bcaa","glutamine"),           # recovery stack
    ("mass gainer","creatine"),
    ("omega 3","protein"),          # joints + muscle recovery
    ("vitamin d","protein"),        # bone + muscle health
    ("zinc","magnesium"),           # ZMA recovery stack
    ("protein","glutamine"),
    # ── Cross-category pairings ───────────────────────────────────────────────
    ("collagen","biotin"),          # beauty-from-within stack
    ("protein","omega 3"),          # general health + muscle
]

SELL_PROBABILITY = {
    "fast_mover":       0.90,
    "healthy_slow":     0.60,
    "overstocked_slow": 0.40,
    "no_demand_slow":   0.20,
    "dead_stock":       0.05,
}
# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY MAPPING
# ══════════════════════════════════════════════════════════════════════════════
PHARMAPLUS_CAT_MAP = {
    "Skin Care":"Beauty Products","Beauty Care & Cosmetics":"Beauty Products",
    "Hair Care":"Beauty Products","Sunscreens":"Beauty Products",
    "Makeup":"Beauty Products","Fragrance":"Beauty Products",
    "Bath & Body":"Beauty Products",
    "Vitamins & Supplements":"Vitamins & Supplements","Vitamins":"Vitamins & Supplements",
    "Supplements":"Vitamins & Supplements","Immune Support":"Vitamins & Supplements",
    "Bone & Heart Health":"Vitamins & Supplements","Energy & Stress":"Vitamins & Supplements",
    "Body Building":"Body Building","Sports Nutrition":"Body Building",
    "Protein":"Body Building","Pre-Workout":"Body Building",
}
MYDAWA_CAT_MAP = {
    "Beauty":"Beauty Products","Skincare":"Beauty Products","Hair":"Beauty Products",
    "Vitamins & Supplements":"Vitamins & Supplements","Supplements":"Vitamins & Supplements",
    "Sports Nutrition":"Body Building","Body Building":"Body Building",
}
BEAUTY_KW = ["serum","moistur","sunscreen","spf","lotion","shampoo","conditioner",
             "mascara","foundation","lipstick","concealer","toner","cleanser",
             "hair","skin","face","body butter","fragrance","perfume"]
SUPP_KW   = ["vitamin","supplement","collagen","biotin","omega","calcium","zinc",
             "magnesium","probiotic","immune","multivitamin","folic","ferrous"]
BB_KW     = ["whey","protein","creatine","bcaa","pre-workout","mass gainer",
             "amino","glutamine","weight gainer","pre workout"]

def classify_by_name(name):
    n = str(name).lower()
    if any(k in n for k in BB_KW):     return "Body Building"
    if any(k in n for k in SUPP_KW):   return "Vitamins & Supplements"
    if any(k in n for k in BEAUTY_KW): return "Beauty Products"
    return None

# ══════════════════════════════════════════════════════════════════════════════
# LOAD PRODUCT LISTS
# ══════════════════════════════════════════════════════════════════════════════
def load_pharmaplus(path):
    df = pd.read_csv(path)
    df["our_category"] = df["category"].map(PHARMAPLUS_CAT_MAP)
    unmap = df["our_category"].isna()
    df.loc[unmap,"our_category"] = df.loc[unmap,"category2"].map(PHARMAPLUS_CAT_MAP)
    df = df[df["our_category"].notna()].copy()
    df = df.rename(columns={"name":"product_name","price_kes":"price","brand":"brand_name"})
    df["units_sold"]      = pd.to_numeric(df.get("units_sold"),      errors="coerce").fillna(0)
    df["in_stock"]        = pd.to_numeric(df.get("in_stock"),        errors="coerce").fillna(0)
    df["is_best_seller"]  = pd.to_numeric(df.get("is_best_seller"),  errors="coerce").fillna(0)
    df["is_on_promotion"] = pd.to_numeric(df.get("is_on_promotion"), errors="coerce").fillna(0)
    df["discount_pct"]    = pd.to_numeric(df.get("promo_discount_value%"), errors="coerce").fillna(0)
    stock_proxy = df["in_stock"].clip(upper=200) * 2
    df["units_sold"] = df["units_sold"].where(df["units_sold"] > 0, stock_proxy)
    return df[["product_name","brand_name","our_category","price",
               "units_sold","is_best_seller","is_on_promotion","discount_pct"]
              ].drop_duplicates("product_name")

def load_goodlife(path):
    df = pd.read_csv(path)
    df["our_category"]    = df["product_name"].apply(classify_by_name)
    df = df[df["our_category"].notna()].copy()
    df = df.rename(columns={"current_price":"price"})
    df["brand_name"]      = ""
    df["units_sold"]      = 0
    df["is_best_seller"]  = 0
    df["is_on_promotion"] = df["discount_badge"].notna().astype(int)
    df["discount_pct"]    = pd.to_numeric(
        df["discount_badge"].str.extract(r"(\d+)")[0], errors="coerce").fillna(0)
    return df[["product_name","brand_name","our_category","price",
               "units_sold","is_best_seller","is_on_promotion","discount_pct"]
              ].drop_duplicates("product_name")

def load_mydawa(path):
    df = pd.read_csv(path)
    df["our_category"] = df["main_category"].map(MYDAWA_CAT_MAP)
    df = df[df["our_category"].notna()].copy()
    df = df.rename(columns={"name":"product_name","price_kes":"price"})
    stock = pd.to_numeric(df.get("stock_qty"), errors="coerce").fillna(0)
    df["units_sold"]      = stock.clip(upper=200) * 2
    df["is_best_seller"]  = 0
    df["is_on_promotion"] = 0
    df["discount_pct"]    = 0
    return df[["product_name","brand_name","our_category","price",
               "units_sold","is_best_seller","is_on_promotion","discount_pct"]
              ].drop_duplicates("product_name")

def load_linton(path):
    df = pd.read_csv(path)
    df["our_category"] = df["product_name"].apply(classify_by_name)
    df = df[df["our_category"].notna()].copy()
    df = df.rename(columns={"current_price":"price"})
    df["brand_name"]      = ""
    df["is_best_seller"]  = 0
    df["is_on_promotion"] = 0
    df["discount_pct"]    = pd.to_numeric(df.get("discount_percentage"), errors="coerce").fillna(0)
    rating  = pd.to_numeric(df.get("rating"),  errors="coerce").fillna(0)
    reviews = pd.to_numeric(df.get("reviews"), errors="coerce").fillna(0)
    df["units_sold"] = (rating * reviews).clip(upper=500)
    return df[["product_name","brand_name","our_category","price",
               "units_sold","is_best_seller","is_on_promotion","discount_pct"]
              ].drop_duplicates("product_name")

print("Loading product lists...")
pp = load_pharmaplus(PHARMAPLUS_PATH)
gl = load_goodlife(GOODLIFE_PATH)
md = load_mydawa(MYDAWA_PATH)
ln = load_linton(LINTON_PATH)

all_products    = pd.concat([pp, gl, md, ln], ignore_index=True)
price_consensus = (all_products.groupby("product_name")["price"]
                   .median().reset_index()
                   .rename(columns={"price":"consensus_price"}))

product_list = pd.concat([
    pp,
    gl[~gl["product_name"].isin(pp["product_name"])],
    md[~md["product_name"].isin(pp["product_name"]) &
       ~md["product_name"].isin(gl["product_name"])],
    ln[~ln["product_name"].isin(pp["product_name"]) &
       ~ln["product_name"].isin(gl["product_name"]) &
       ~ln["product_name"].isin(md["product_name"])],
], ignore_index=True).drop_duplicates("product_name")

product_list = product_list.merge(price_consensus, on="product_name", how="left")
product_list["price"] = product_list["consensus_price"].fillna(product_list["price"])
product_list = product_list.drop(columns=["consensus_price"]).reset_index(drop=True)
product_list["product_id"] = product_list.index + 90001

for col in ["units_sold","is_best_seller","is_on_promotion","discount_pct"]:
    product_list[col] = pd.to_numeric(product_list.get(col), errors="coerce").fillna(0)

product_list.to_csv(os.path.join(r"C:\Users\Mercy\Documents\Tendri\Snowflake Pulls\Xana\snowflake\tendri\unified_data", f"unified_product_list_{TODAY}.csv"), index=False)

print(f"Unified product list: {len(product_list):,} products")
for cat, grp in product_list.groupby("our_category"):
    print(f"  {cat:<30} {len(grp):,} products")

product_list["is_generic"]        = product_list.apply(
    lambda r: detect_generic(r["product_name"], r.get("brand_name","")), axis=1)
product_list["repurchase_cycle"]  = product_list["product_name"].apply(get_repurchase_cycle)

product_names_lower = set(product_list["product_name"].str.lower())

def has_basket_partner(product_name):
    name = str(product_name).lower()
    for a, b in BASKET_PAIRS:
        if a in name and any(b in other for other in product_names_lower):
            return True
        if b in name and any(a in other for other in product_names_lower):
            return True
    return False

product_list["has_basket_partner"] = product_list["product_name"].apply(has_basket_partner)

# ══════════════════════════════════════════════════════════════════════════════
# ASSIGN INVENTORY HEALTH STATES
# ══════════════════════════════════════════════════════════════════════════════
def assign_health_states(df):
    result = []
    for cat, grp in df.groupby("our_category"):
        grp = grp.copy().reset_index(drop=True)
        n   = len(grp)
        states = []
        for state, cfg in HEALTH_PROFILES.items():
            count = max(1, round(n * cfg["pct"]))
            states.extend([state] * count)
        states = (states * ((n // len(states)) + 1))[:n]
        np.random.shuffle(states)
        grp["inventory_health"] = states
        result.append(grp)
    return pd.concat(result, ignore_index=True)

product_list = assign_health_states(product_list)


# ── Cap products per category to realistic pharmacy SKU counts ────────────
CATEGORY_SKU_CAP = {
    "Beauty Products":        180,
    "Vitamins & Supplements": 180,
    "Body Building":           80,
}

capped = []
for cat, cap in CATEGORY_SKU_CAP.items():
    grp = product_list[product_list["our_category"] == cat].copy()
    if len(grp) > cap:
        # Prioritise best sellers, then sample the rest
        best = grp[grp["is_best_seller"] == 1]
        rest = grp[grp["is_best_seller"] == 0].sample(
            min(cap - len(best), len(rest := grp[grp["is_best_seller"] == 0])),
            random_state=42
        )
        grp = pd.concat([best, rest], ignore_index=True).head(cap)
    capped.append(grp)

product_list = pd.concat(capped, ignore_index=True).reset_index(drop=True)
product_list["product_id"] = product_list.index + 90001  # reassign IDs

print("Product list after capping:")
for cat, grp in product_list.groupby("our_category"):
    print(f"  {cat:<30} {len(grp):,} products")


# ══════════════════════════════════════════════════════════════════════════════
# BASE MONTHLY VELOCITY
# ══════════════════════════════════════════════════════════════════════════════
VELOCITY_ANCHORS = {
    "Beauty Products":        25,
    "Vitamins & Supplements": 18,
    "Body Building":           8,
}
MEDIAN_PRICE = 1575

def base_velocity(row):
    cat    = row["our_category"]
    price  = max(row["price"], 50)
    sold   = row["units_sold"]
    anchor = VELOCITY_ANCHORS.get(cat, 15)
    health = row["inventory_health"]

    # Signal priority: actual sales > stock proxy > price-derived
    if sold > 0:
        base = max(sold / 12, 1.0)
    else:
        base = anchor * ((MEDIAN_PRICE / price) ** 0.5)

    # Inventory health state scale
    health_scale = {
        "fast_mover": 2.5, "healthy_slow": 0.8,
        "overstocked_slow": 0.4, "no_demand_slow": 0.15, "dead_stock": 0.05
    }
    base *= health_scale.get(health, 1.0)

    # Brand tier (budget moves faster, premium slower)
    base *= BRAND_TIER_MULT.get(margin_tier(price), 1.0)

    # Repurchase frequency (sunscreen monthly, lipstick biannual)
    base *= REPURCHASE_VELOCITY_MULT.get(row.get("repurchase_cycle","quarterly"), 0.50)

    # Pharmacist uplift by category
    base *= PHARMACIST_UPLIFT.get(cat, 1.0)

    # Prescription spillover (+20% for adjacent pharma products)
    name = str(row["product_name"]).lower()
    if any(kw in name for kw in SPILLOVER_KEYWORDS):
        base *= 1.20

    # Impulse purchase (+12% for products < KES 500)
    if price < IMPULSE_MAX_PRICE:
        base *= 1.12

    # Basket coupling (+15% if a common pair partner exists)
    if row.get("has_basket_partner", False):
        base *= 1.15

    # Bestseller (+30%)
    if row.get("is_best_seller", 0):
        base *= 1.30

    # Discount elasticity: each 10% discount -> +5% volume
    disc = row.get("discount_pct", 0)
    if disc > 0:
        base *= (1 + (disc / 10) * 0.05)

    return max(round(base, 2), 0.1)

product_list["monthly_velocity"] = product_list.apply(base_velocity, axis=1)

# Scale to hit category targets (anchored to pharma mix)
for cat, target in CATEGORY_TARGETS.items():
    mask = product_list["our_category"] == cat
    current_total = product_list.loc[mask, "monthly_velocity"].sum()
    if current_total > 0:
        scale = target / current_total
        product_list.loc[mask, "monthly_velocity"] = (
            product_list.loc[mask, "monthly_velocity"] * scale
        ).round(2)

print("\nCategory volume targets vs achieved:")
for cat, target in CATEGORY_TARGETS.items():
    achieved = product_list[product_list["our_category"]==cat]["monthly_velocity"].sum()
    print(f"  {cat:<30} target={target:>8,.0f}  achieved={achieved:>8,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD GOOGLE TRENDS INDEX
# ══════════════════════════════════════════════════════════════════════════════
try:
    _gt  = sorted(glob.glob(EMBU_GT_PATH))[-1]
    embu_gt = (pd.read_csv(_gt, parse_dates=["month"])
               .set_index(["month","category"])["category_index"])
    print(f"\nLoaded Embu GT index: {_gt}")
except (IndexError, FileNotFoundError):
    embu_gt = None
    print("\nNo Embu GT index — using seasonal patterns only")

def gt_scalar(month_ts, category):
    """
    Floor 45% + GT signal x 40% = range 0.45 to 0.85
    Prevents any month collapsing to near-zero.
    """
    if embu_gt is None:
        return 0.60
    key  = (pd.Timestamp(month_ts), category)
    if key in embu_gt.index:
        raw = float(embu_gt[key])
    else:
        vals = (embu_gt.xs(category, level="category")
                if category in embu_gt.index.get_level_values("category") else None)
        raw  = float(vals.mean()) if vals is not None else 0.60
    return 0.45 + (raw * 0.40)

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATE DISPENSING
# ══════════════════════════════════════════════════════════════════════════════
months = pd.date_range(start=SIM_START, end=SIM_END, freq="MS")
print(f"\nSimulating {len(months)} months ({SIM_START} to {SIM_END})")
print(f"Products: {len(product_list):,} | Facilities: {FACILITY_IDS}")

agg_rows  = []
fact_rows = []

for facility_id in FACILITY_IDS:
    vol_split    = FACILITY_VOLUME_SPLIT[facility_id]
    patient_base = FACILITY_PATIENT_BASE[facility_id]
    print(f"\n  Facility {facility_id} ({int(vol_split*100)}% of volume, {patient_base:,} patient base)...")

    for i, month in enumerate(months):
        month_rank = i + 1
        month_num  = month.month

        for _, prod in product_list.iterrows():
            cat      = prod["our_category"]
            health   = prod["inventory_health"]
            velocity = prod["monthly_velocity"] * vol_split

            if health == "dead_stock":
                qty = max(int(np.random.poisson(0.3)), 0)
                if qty == 0:
                    continue
            else:
                ramp     = STORE_MATURITY[cat].get(month_rank, 1.0)
                seasonal = SEASONAL[cat][month_num]
                gt       = gt_scalar(month, cat)
                habit    = (HABIT.get(month_rank, 1.0)
                            if cat == "Vitamins & Supplements" else 1.0)
                promo    = promo_mult(prod.get("is_on_promotion", 0), month)

                # GT slope: bestsellers trend up, no-demand products trend down
                slope_mult = 1.0
                if prod.get("is_best_seller", 0) and month_rank > 6:
                    slope_mult = GT_SLOPE_BOOST
                elif health in ("no_demand_slow","dead_stock") and month_rank > 6:
                    slope_mult = GT_SLOPE_PENALTY

                # BB income sensitivity:
                # Payday boost every month (first-week effect) +
                # Additional bonus month boost in Jan, Apr, Aug, Dec
                income_mult = 1.0
                if cat == "Body Building":
                    income_mult = BB_PAYDAY_BOOST  # base payday effect every month
                    if month_num in BB_BONUS_MONTHS:
                        income_mult *= BB_BONUS_MONTHS[month_num]  # stack bonus on top
                
                sell_prob = SELL_PROBABILITY.get(health, 0.60)
                if np.random.random() > sell_prob:
                    continue  # product doesn't sell this month

                expected = velocity * ramp * seasonal * gt * habit * promo * slope_mult * income_mult

                shape = 25
                qty   = max(int(np.random.gamma(shape, max(expected / shape, 0.01))), 0)
                if qty == 0:
                    continue

            price    = max(prod["price"], 1)
            patients = compute_unique_patients(qty, prod["repurchase_cycle"])

            # monthly_dispensing_aggregate row (matches schema exactly)
            agg_rows.append({
                "facility_id"              : facility_id,
                "months"                   : month.strftime("%Y-%m-%d"),
                "product_id"               : prod["product_id"],
                "new_category_name"        : cat,
                "parent_category_name"     : cat,
                "correct_therapeutic_class": cat,
                "total_qty_dispensed"      : qty,
                "unique_patients"          : patients,
                "avg_qty_per_patient"      : round(qty / patients, 2),
            })

            # fact_dispensing rows (no store_id)
            n_events    = max(1, qty // 3)
            dates_in_month = pd.date_range(
                start=month, end=month + pd.offsets.MonthEnd(0), freq="B"
            )
            event_dates   = np.random.choice(
                dates_in_month, size=min(n_events, len(dates_in_month)), replace=True
            )
            qty_remaining = qty
            for evt_date in event_dates:
                if qty_remaining <= 0:
                    break
                evt_qty       = min(max(int(np.random.poisson(3)), 1), qty_remaining)
                qty_remaining -= evt_qty
                patient_id    = f"P{facility_id}_{np.random.randint(1, patient_base):06d}"

                if health == "dead_stock":
                    # Use category-specific dead stock threshold
                    # so the dashboard's detection logic fires correctly per category
                    threshold = DEAD_STOCK_THRESHOLD_DAYS.get(cat, 120)
                    evt_date = (month - pd.Timedelta(
                        days=threshold + np.random.randint(0, 30)
                    )).date()

                fact_rows.append({
                    "date"              : str(evt_date),
                    "facility_id"       : facility_id,
                    "patient_id"        : patient_id,
                    "product_id"        : prod["product_id"],
                    "product_name"      : prod["product_name"],
                    "qty_dispensed"     : evt_qty,
                    "unit_selling_price": round(price, 2),
                    "total_sales_value" : round(price * evt_qty, 2),
                    "category"          : cat,
                    "inventory_health"  : health,
                })

agg_df  = pd.DataFrame(agg_rows)
fact_df = pd.DataFrame(fact_rows)
print(f"\n  Dispensing aggregate: {len(agg_df):,} rows")
print(f"  Fact dispensing:      {len(fact_df):,} rows")

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATE INVENTORY SNAPSHOT (no store_id)
# Overstocked slow: stock_ratio = 6.0 (too much ordered -> purchasing problem)
# No-demand slow:   stock_ratio = 1.5 (normal order, product just doesn't sell)
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating inventory snapshots...")
snap_rows = []

for facility_id in FACILITY_IDS:
    vol_split = FACILITY_VOLUME_SPLIT[facility_id]
    for i, month in enumerate(months):
        for _, prod in product_list.iterrows():
            health      = prod["inventory_health"]
            velocity    = prod["monthly_velocity"] * vol_split
            price       = max(prod["price"], 1)
            stock_ratio = HEALTH_PROFILES[health]["stock_ratio"]

            base_stock  = velocity * stock_ratio
            noise       = np.random.uniform(0.85, 1.15)
            qty_on_hand = max(round(base_stock * noise), 0)

            reorder_level       = max(round(velocity * 0.5), 1)
            smart_reorder_level = max(round(velocity * 0.75), 1)
            is_stockout         = int(qty_on_hand == 0)
            is_low_stock        = int(0 < qty_on_hand <= reorder_level)
            unit_cost           = round(price * np.random.uniform(0.55, 0.75), 2)
            inv_value           = round(qty_on_hand * unit_cost, 2)

            snap_rows.append({
                "snapshot_id"          : str(uuid.uuid4()),
                "snapshot_date"        : month.strftime("%Y-%m-%d"),
                "facility_id"          : facility_id,
                "product_id"           : prod["product_id"],
                "product_name"         : prod["product_name"],
                "qty_on_hand"          : qty_on_hand,
                "smart_reorder_level"  : smart_reorder_level,
                "re_order_level"       : reorder_level,
                "is_stockout"          : is_stockout,
                "is_low_stock"         : is_low_stock,
                "unit_cost"            : unit_cost,
                "total_inventory_value": inv_value,
            })

snap_df = pd.DataFrame(snap_rows)
print(f"  Inventory snapshot: {len(snap_df):,} rows")

# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT INTELLIGENCE TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding product intelligence table...")
intel_rows = []

for cat, grp in product_list.groupby("our_category"):
    grp = grp.copy()
    for _, prod in grp.iterrows():
        gm             = gross_margin(prod["price"], prod["is_generic"])
        monthly_vol    = prod["monthly_velocity"]
        margin_contrib = round(monthly_vol * prod["price"] * gm, 2)
        saving, generic_equiv = generic_opportunity(prod, grp)

        health = prod["inventory_health"]
        if health == "fast_mover":
            intervention = "Maintain stock — ensure no stockout"
        elif health == "healthy_slow":
            intervention = "Monitor — normal slow mover"
        elif health == "overstocked_slow":
            intervention = "Reduce order qty — transfer excess or bundle with fast mover"
        elif health == "no_demand_slow":
            intervention = "Review ranging — consider delisting or cross-branch transfer"
        else:
            intervention = "Clearance — apply deep discount or return to supplier"

        dos = round(
            (prod["monthly_velocity"] * HEALTH_PROFILES[health]["stock_ratio"] * 30)
            / max(prod["monthly_velocity"], 0.1), 0
        )

        intel_rows.append({
            "product_id"                     : prod["product_id"],
            "product_name"                   : prod["product_name"],
            "brand_name"                     : prod.get("brand_name",""),
            "category"                       : cat,
            "price_kes"                      : round(prod["price"], 2),
            "margin_tier"                    : margin_tier(prod["price"]),
            "gross_margin_pct"               : gm,
            "monthly_velocity"               : round(prod["monthly_velocity"], 1),
            "monthly_margin_contribution_kes": margin_contrib,
            "inventory_health"               : health,
            "slow_mover_cause"               : (
                "overstocking" if health == "overstocked_slow"
                else "no_demand" if health in ("no_demand_slow","dead_stock")
                else None
            ),
            "days_of_stock"                  : dos,
            "dead_stock_threshold_days"      : DEAD_STOCK_THRESHOLD_DAYS.get(cat, 120),
            "repurchase_cycle"               : prod["repurchase_cycle"],
            "is_generic"                     : prod["is_generic"],
            "is_best_seller"                 : bool(prod.get("is_best_seller",0)),
            "generic_saving_per_month_kes"   : saving,
            "generic_equivalent"             : generic_equiv,
            "intervention"                   : intervention,
        })

intel_df = pd.DataFrame(intel_rows)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE ALL OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
outputs = {
    f"simulated_dispensing_aggregate_{TODAY}.csv" : agg_df,
    f"simulated_fact_dispensing_{TODAY}.csv"      : fact_df,
    f"simulated_inventory_snapshot_{TODAY}.csv"   : snap_df,
    f"simulated_product_intelligence_{TODAY}.csv" : intel_df,
}

print("\n" + "="*60)
for fname, df in outputs.items():
    path = os.path.join(OUT_DIR, fname)
    df.to_csv(path, index=False)
    print(f"  v {fname}  ({len(df):,} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("\nVALIDATION")
print("="*60)

core_cols = ["facility_id","months","product_id","new_category_name",
             "parent_category_name","correct_therapeutic_class",
             "total_qty_dispensed","unique_patients","avg_qty_per_patient"]
missing = [c for c in core_cols if c not in agg_df.columns]
print(f"  {'OK' if not missing else 'FAIL'} Schema: {missing if missing else 'all columns present'}")

for name, df in outputs.items():
    has_store = "store_id" in df.columns
    print(f"  {'FAIL store_id present' if has_store else 'OK no store_id'} — {name}")

print("\nCategory volume vs targets (monthly avg across both facilities):")
for cat, target in CATEGORY_TARGETS.items():
    actual = agg_df[agg_df["new_category_name"]==cat]["total_qty_dispensed"].sum() / len(months)
    pct    = actual / target * 100
    print(f"  {cat:<30} target={target:>8,.0f}  actual={actual:>8,.0f}  ({pct:.1f}%)")

print("\nInventory health distribution:")
hc = intel_df["inventory_health"].value_counts()
for state, count in hc.items():
    pct = count / len(intel_df) * 100
    print(f"  {state:<20} {count:>4} products ({pct:.1f}%)")

slow = intel_df[intel_df["slow_mover_cause"].notna()]
print(f"\nSlow mover cause split ({len(slow)} products):")
print(f"  Overstocking: {(slow['slow_mover_cause']=='overstocking').sum()}")
print(f"  No demand:    {(slow['slow_mover_cause']=='no_demand').sum()}")

print(f"\nGeneric vs branded:")
print(f"  Generic:  {intel_df['is_generic'].sum()}")
print(f"  Branded:  {(~intel_df['is_generic']).sum()}")
sav = intel_df[intel_df["generic_saving_per_month_kes"]>0]
print(f"  Substitution opportunities: {len(sav)} products")
print(f"  Total potential saving: KES {sav['generic_saving_per_month_kes'].sum():,.0f}/month")

print("\nMonthly margin contribution by category:")
mc = intel_df.groupby("category")["monthly_margin_contribution_kes"].sum().sort_values(ascending=False)
for cat, contrib in mc.items():
    print(f"  {cat:<30} KES {contrib:>12,.0f}/month")

print(f"\nunique_patients: min={agg_df['unique_patients'].min()}  max={agg_df['unique_patients'].max()}")
print(f"avg_qty_per_patient: min={agg_df['avg_qty_per_patient'].min():.1f}  max={agg_df['avg_qty_per_patient'].max():.1f}")

print("\nAll done.")

# ══════════════════════════════════════════════════════════════════════════════
# MERGE SNIPPET FOR KNN NOTEBOOK
# ══════════════════════════════════════════════════════════════════════════════
print("""
NOTEBOOK MERGE SNIPPET
======================
import glob, pandas as pd

_path = sorted(glob.glob("tendri/data/simulated_dispensing_aggregate_*.csv"))[-1]
sim_agg = pd.read_csv(_path)
sim_agg["months"] = pd.to_datetime(sim_agg["months"])

monthly_dispensing_aggregate = pd.concat(
    [monthly_dispensing_aggregate, sim_agg], ignore_index=True
)
print(f"Categories: {monthly_dispensing_aggregate['new_category_name'].unique().tolist()}")
print(f"Total rows: {len(monthly_dispensing_aggregate):,}")


STREAMLIT PICKLE UPDATE
=======================
import glob, pandas as pd, pickle

sim_disp  = pd.read_csv(sorted(glob.glob("tendri/data/simulated_fact_dispensing_*.csv"))[-1], parse_dates=["date"])
sim_snap  = pd.read_csv(sorted(glob.glob("tendri/data/simulated_inventory_snapshot_*.csv"))[-1], parse_dates=["snapshot_date"])
sim_intel = pd.read_csv(sorted(glob.glob("tendri/data/simulated_product_intelligence_*.csv"))[-1])

fact_dispensing         = pd.concat([fact_dispensing, sim_disp],  ignore_index=True)
fact_inventory_snapshot = pd.concat([fact_inventory_snapshot, sim_snap], ignore_index=True)

pickle.dump({
    "disp"         : fact_dispensing,
    "inv"          : fact_inventory_snapshot,
    "pat"          : dim_patient_profile,
    "diag_df"      : monthly_diagnoses_aggregate,
    "disp_df"      : monthly_dispensing_aggregate,
    "pred"         : output,
    "pred_products": output_products,
    "product_intel": sim_intel,
}, open("data_export.pkl", "wb"))
print("Pickle updated.")
""")