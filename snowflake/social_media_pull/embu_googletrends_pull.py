"""
Embu Google Trends Pull — Simulation Signal
=============================================
Pulls monthly Google Trends data for Kenya (geo="KE") using Embu-specific
keywords — e.g. "moisturiser embu" instead of "moisturiser kenya".

This is more reliable than geo="KE-300" which causes a 400 error because
Google Trends does not expose sub-county codes reliably via the API.

Run this script first, then run dispensing_simulation.py.

Outputs:
  embu_trends/
  ├── checkpoints/                          ← one CSV per subgroup
  ├── embu_trends_monthly_DATE.csv
  └── embu_trends_category_index_DATE.csv  ← consumed by simulation
"""

import os
import time
import random
import warnings
import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
TODAY          = datetime.now().strftime("%Y-%m-%d")
TIMEFRAME      = "2023-01-01 " + TODAY
GEO            = "KE"          # Kenya — geo already filters the country
SLEEP_MIN      = 10
SLEEP_MAX      = 20
MAX_RETRY      = 3
RETRY_WAIT     = 120
OUTPUT_DIR     = "embu_trends"
CHECKPOINT_DIR = "embu_trends/checkpoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def random_sleep():
    t = random.uniform(SLEEP_MIN, SLEEP_MAX)
    print(f"    (sleeping {t:.1f}s...)", end=" ", flush=True)
    time.sleep(t)


def build_pytrends():
    """
    Fresh pytrends client.
    retries/backoff_factor omitted — they use urllib3 Retry(method_whitelist=...)
    which was renamed in urllib3 2.0+ and causes a crash. Rate limiting is
    handled manually via random_sleep and the retry loop in fetch_monthly.
    """
    return TrendReq(hl="en-US", tz=-180, timeout=(10, 30))


# ── Checkpoint functions ──────────────────────────────────────────────────────

def checkpoint_path(category, sub_name):
    """Build the file path for a subgroup checkpoint."""
    safe_cat = category.replace(" ", "_").replace("&", "and")
    return f"{CHECKPOINT_DIR}/{safe_cat}__{sub_name}.csv"


def already_done(category, sub_name):
    """Returns True if this subgroup was successfully pulled in a prior run."""
    return os.path.exists(checkpoint_path(category, sub_name))


def save_checkpoint(df_long, category, sub_name):
    """Save subgroup result to disk immediately after a successful pull."""
    df_long.to_csv(checkpoint_path(category, sub_name), index=False)


def load_all_checkpoints():
    """Load every saved checkpoint into one combined DataFrame."""
    files = [
        os.path.join(CHECKPOINT_DIR, f)
        for f in os.listdir(CHECKPOINT_DIR)
        if f.endswith(".csv") and "__" in f
    ]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# FETCH
# ══════════════════════════════════════════════════════════════════════════════

pytrends = build_pytrends()


def fetch_monthly(keywords, cat, attempt=1):
    """
    Pull monthly interest_over_time for Kenya.
    geo="KE" already scopes results to Kenya — no location suffix needed
    in the keywords. Plain terms like "moisturiser" return real volume.
    Retries on 429 with escalating back-off. Returns DataFrame or empty.
    """
    global pytrends

    try:
        pytrends.build_payload(
            kw_list=keywords, geo=GEO,
            timeframe=TIMEFRAME, cat=cat, gprop=""
        )
        random_sleep()
        df = pytrends.interest_over_time()
        if df.empty:
            return pd.DataFrame()
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        return df

    except TooManyRequestsError:
        if attempt <= MAX_RETRY:
            print(f"\n  ⚠️  429 received.")
            print(f"  Tip: switch to a hotspot or VPN now for a fresh IP,")
            print(f"       then re-run — checkpoints will resume from here.")
            print(f"  Waiting {RETRY_WAIT * attempt}s before auto-retry "
                  f"({attempt}/{MAX_RETRY})...")
            time.sleep(RETRY_WAIT * attempt)
            pytrends = build_pytrends()
            return fetch_monthly(keywords, cat, attempt + 1)
        print(f"\n  ✗  429 persists after {MAX_RETRY} retries.")
        print(f"  Switch networks and re-run to resume from checkpoint.")
        return pd.DataFrame()

    except Exception as e:
        print(f"\n  ✗  Error: {e} — skipping")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD GROUPS
# ══════════════════════════════════════════════════════════════════════════════

KEYWORD_GROUPS = {

    "Beauty Products": {
        "cat": 44,
        "subgroups": {
            "skincare_brands_1": [
                "Nivea", "Neutrogena", "CeraVe", "COSRX", "La Roche-Posay",
            ],
            "skincare_brands_2": [
                "EOS", "Olay", "The Ordinary", "Bioderma", "Vaseline",
            ],
            "skincare_products_1": [
                "face serum", "moisturiser", "toner", "body lotion", "face wash",
            ],
            "skincare_products_2": [
                "vitamin C serum", "retinol cream", "hyaluronic acid",
                "niacinamide serum", "exfoliator",
            ],
            "haircare_brands_1": [
                "Dark and Lovely", "Cantu", "ORS", "Sunsilk", "Dove shampoo",
            ],
            "haircare_brands_2": [
                "Schwarzkopf", "Pantene", "TRESemmé",
                "Head and Shoulders", "Garnier hair",
            ],
            "haircare_products_1": [
                "hair relaxer", "hair growth oil", "deep conditioner",
                "natural hair products", "anti dandruff shampoo",
            ],
            "haircare_products_2": [
                "hair mask", "leave in conditioner", "edge control",
                "heat protectant", "hair serum",
            ],
            "beauty_cosmetics_brands_1": [
                "Maybelline", "Revlon", "Black Opal", "Wet n Wild", "Ruby Rose",
            ],
            "beauty_cosmetics_brands_2": [
                "e.l.f cosmetics", "Milani", "LA Girl", "Catrice",
                "essence cosmetics",
            ],
            "makeup_brands_1": [
                "NYX cosmetics", "MAC cosmetics", "Sleek makeup",
                "Rimmel", "Flormar",
            ],
            "makeup_brands_2": [
                "Charlotte Tilbury", "Urban Decay", "Too Faced",
                "NARS", "Fenty Beauty",
            ],
            "makeup_products_1": [
                "foundation", "lipstick", "mascara",
                "setting powder", "concealer",
            ],
            "makeup_products_2": [
                "eyeshadow palette", "blush", "highlighter makeup",
                "lip gloss", "eyeliner",
            ],
            "sunscreen_brands_1": [
                "Neutrogena sunscreen", "Nivea sun", "La Roche-Posay",
                "Bioderma sunscreen", "Garnier sunscreen",
            ],
            "sunscreen_brands_2": [
                "Eucerin sun", "Coppertone", "Ambre Solaire",
                "ISDIN", "Altruist sunscreen",
            ],
            "sunscreen_products_1": [
                "sunscreen", "SPF 50", "tinted sunscreen",
                "daily sunscreen", "sunblock",
            ],
            "sunscreen_products_2": [
                "sunscreen for black skin", "mineral sunscreen",
                "face sunscreen", "body sunscreen", "UV protection cream",
            ],
        }
    },

    "Vitamins & Supplements": {
        "cat": 44,
        "subgroups": {
            "immune_brands_1": [
                "Redoxon", "Berocca", "Ester-C", "Zinnat", "Supavit",
            ],
            "immune_brands_2": [
                "Emergen-C", "Nature's Bounty vitamin C",
                "Solgar vitamin C", "Blackmores vitamin C",
                "Holland and Barrett vitamin C",
            ],
            "immune_products_1": [
                "Vitamin C supplement", "zinc supplement",
                "effervescent vitamin C", "immune booster", "vitamin C 1000mg",
            ],
            "immune_products_2": [
                "vitamin C tablets", "zinc and vitamin C",
                "vitamin C powder", "immune support supplement",
                "elderberry supplement",
            ],
            "multivitamin_brands_1": [
                "Centrum", "Seven Seas", "Supradyn", "Vitabiotics", "Abidec",
            ],
            "multivitamin_brands_2": [
                "Blackmores", "Solgar multivitamin", "Nature Made",
                "Garden of Life", "Kirkland vitamins",
            ],
            "multivitamin_products_1": [
                "multivitamin", "prenatal vitamins", "Wellwoman",
                "Wellman", "children multivitamin",
            ],
            "multivitamin_products_2": [
                "mens multivitamin", "womens multivitamin",
                "senior multivitamin", "gummy vitamins", "daily vitamins",
            ],
            "beauty_supplement_brands_1": [
                "Perfectil", "Nourkrin", "NeoCell",
                "Nature's Bounty biotin", "Zeta White",
            ],
            "beauty_supplement_brands_2": [
                "Vital Proteins", "Sports Research collagen",
                "Further Food collagen", "Ancient Nutrition",
                "Garden of Life collagen",
            ],
            "beauty_supplement_products_1": [
                "collagen supplement", "biotin supplement", "marine collagen",
                "collagen powder", "hair skin nails supplement",
            ],
            "beauty_supplement_products_2": [
                "collagen tablets", "biotin 10000mcg", "collagen drink",
                "beauty vitamins", "keratin supplement",
            ],
            "bone_heart_brands_1": [
                "Seven Seas Cod Liver Oil", "Omega H3", "Calcichew",
                "Caltrate", "Cardiowell",
            ],
            "bone_heart_brands_2": [
                "Blackmores fish oil", "Solgar omega 3", "Nature Made fish oil",
                "Nordic Naturals", "Kirkland fish oil",
            ],
            "bone_heart_products_1": [
                "Vitamin D supplement", "calcium supplement",
                "omega 3 fish oil", "cod liver oil", "Calcium D3",
            ],
            "bone_heart_products_2": [
                "vitamin D3", "calcium magnesium zinc", "fish oil capsules",
                "heart health supplement", "bone supplement",
            ],
            "energy_stress_brands_1": [
                "Neurobion", "Becosules", "Slow-Mag",
                "Magnesium B6", "Berocca Performance",
            ],
            "energy_stress_brands_2": [
                "Solgar B complex", "Nature Made B12", "Blackmores B complex",
                "Metagenics magnesium", "Pure Encapsulations",
            ],
            "energy_stress_products_1": [
                "B complex vitamin", "magnesium supplement",
                "energy supplement", "stress relief supplement", "vitamin B12",
            ],
            "energy_stress_products_2": [
                "magnesium glycinate", "ashwagandha", "B12 injection",
                "adaptogen supplement", "fatigue supplement",
            ],
        }
    },

    "Body Building": {
        "cat": 44,
        "subgroups": {
            "protein_brands_1": [
                "Optimum Nutrition", "USN protein", "BSN Syntha-6",
                "Muscletech", "Evox protein",
            ],
            "protein_brands_2": [
                "Dymatize protein", "Isopure", "MyProtein",
                "MuscleMeds", "Rule 1 protein",
            ],
            "protein_products_1": [
                "whey protein", "protein powder", "isolate protein",
                "plant protein", "protein shake",
            ],
            "protein_products_2": [
                "casein protein", "egg white protein", "protein bar",
                "whey concentrate", "vegan protein powder",
            ],
            "mass_gainer_brands_1": [
                "Serious Mass", "USN Muscle Fuel", "Dymatize Super Mass",
                "Mutant Mass", "Evox Mass",
            ],
            "mass_gainer_brands_2": [
                "Optimum Nutrition mass", "BSN True Mass",
                "Muscletech mass gainer", "MyProtein mass gainer", "Naked Mass",
            ],
            "mass_gainer_products_1": [
                "mass gainer", "weight gainer", "bulk supplement",
                "mass gainer chocolate", "3000 calorie shake",
            ],
            "mass_gainer_products_2": [
                "high calorie supplement", "hardgainer supplement",
                "mass gainer vanilla", "lean mass gainer", "mass gainer price",
            ],
            "creatine_brands_1": [
                "Creapure", "Optimum Nutrition creatine", "USN creatine",
                "Muscletech creatine", "BPI creatine",
            ],
            "creatine_brands_2": [
                "Kaged creatine", "Klean Athlete creatine",
                "Bulk Powders creatine", "MyProtein creatine", "Allmax creatine",
            ],
            "creatine_products_1": [
                "creatine", "creatine monohydrate", "creatine powder",
                "creatine supplement", "gym creatine",
            ],
            "creatine_products_2": [
                "creatine HCL", "creatine loading", "creatine capsules",
                "creatine and protein", "creatine for women",
            ],
            "preworkout_brands_1": [
                "C4 pre workout", "NO Xplode", "Ghost pre workout",
                "USN 3XT", "Muscletech Nano X",
            ],
            "preworkout_brands_2": [
                "Kaged Pre-Kaged", "Gorilla Mode", "Total War pre workout",
                "Wrecked pre workout", "Bucked Up",
            ],
            "preworkout_products_1": [
                "pre workout", "energy booster gym", "pre workout powder",
                "caffeine pre workout", "beta alanine",
            ],
            "preworkout_products_2": [
                "stim free pre workout", "pre workout drink", "pump supplement",
                "nitric oxide supplement", "citrulline supplement",
            ],
            "amino_brands_1": [
                "Scivation Xtend", "Optimum Nutrition BCAA", "USN BCAA",
                "Evox BCAA", "Muscletech amino",
            ],
            "amino_brands_2": [
                "Kaged BCAA", "Cellucor BCAA", "MyProtein BCAA",
                "BSN amino", "Allmax BCAA",
            ],
            "amino_products_1": [
                "BCAA", "EAA supplement", "amino acids",
                "glutamine supplement", "post workout supplement",
            ],
            "amino_products_2": [
                "BCAA powder", "recovery supplement", "BCAA capsules",
                "intra workout supplement", "electrolyte supplement",
            ],
        }
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PULL
# ══════════════════════════════════════════════════════════════════════════════

total_subs = sum(len(cfg["subgroups"]) for cfg in KEYWORD_GROUPS.values())
done_count = sum(
    already_done(cat, sub)
    for cat, cfg in KEYWORD_GROUPS.items()
    for sub in cfg["subgroups"]
)

print("=" * 65)
print(f"  Embu Google Trends Pull  |  geo={GEO}")
print(f"  Keywords: ' kenya' stripped — geo scopes country automatically")
print(f"  Period  : {TIMEFRAME}")
print(f"  Delays  : {SLEEP_MIN}–{SLEEP_MAX}s  |  "
      f"Retry wait: {RETRY_WAIT}s  |  Max retries: {MAX_RETRY}")
print(f"  Progress: {done_count}/{total_subs} subgroups already done")
print("=" * 65)

completed = []
skipped   = []
seq       = 0

for category, cfg in KEYWORD_GROUPS.items():
    cat       = cfg["cat"]
    subgroups = cfg["subgroups"]
    print(f"\n[{category.upper()}]")

    for sub_name, keywords in subgroups.items():
        seq += 1
        label = f"[{seq}/{total_subs}] {sub_name}"

        if already_done(category, sub_name):
            print(f"  {label} — ✓ already done (checkpoint exists)")
            completed.append(f"{category} / {sub_name}")
            continue

        print(f"  {label}...", end=" ", flush=True)
        df = fetch_monthly(keywords, cat)

        if df.empty:
            skipped.append(f"{category} / {sub_name}")
            print("✗  no data or persistent 429")
            continue

        df = df.reset_index().rename(columns={"date": "month"})
        kw_cols = [c for c in df.columns if c != "month"]
        df_long = df.melt(
            id_vars    = ["month"],
            value_vars = kw_cols,
            var_name   = "keyword",
            value_name = "interest_score"
        )
        df_long["category"] = category
        df_long["subgroup"] = sub_name

        save_checkpoint(df_long, category, sub_name)
        completed.append(f"{category} / {sub_name}")
        print(f"✓  {df_long['month'].nunique()} months — checkpoint saved")

print(f"\n{'='*65}")
print(f"Session complete — {len(completed)} done, {len(skipped)} skipped")
if skipped:
    print("Skipped (re-run to retry):")
    for s in skipped:
        print(f"  - {s}")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD CATEGORY INDEX FROM ALL CHECKPOINTS
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding category index from all checkpoints...")
raw_df = load_all_checkpoints()

if raw_df.empty:
    raise RuntimeError(
        "No checkpoint data found. All subgroups returned empty or were "
        "blocked. Wait 30+ minutes then re-run."
    )

raw_df["month"] = pd.to_datetime(raw_df["month"])

# Normalise each keyword to 0-1 across its full history so high-volume
# keywords don't dominate the category average
raw_df["normalised_score"] = raw_df.groupby("keyword")["interest_score"].transform(
    lambda x: (x / x.max()).round(4) if x.max() > 0 else x
)

# Category index: mean normalised score across all subgroups per month
category_index = (
    raw_df.groupby(["month", "category"])["normalised_score"]
    .mean()
    .reset_index()
    .rename(columns={"normalised_score": "category_index"})
    .round(4)
)

# Fill any missing months via forward/back fill within each category
all_months = pd.date_range(
    start=TIMEFRAME.split(" ")[0], end=TODAY, freq="MS"
)
full_index = pd.MultiIndex.from_product(
    [all_months, list(KEYWORD_GROUPS.keys())],
    names=["month", "category"]
)
category_index = (
    category_index
    .set_index(["month", "category"])
    .reindex(full_index)
    .reset_index()
)
category_index["category_index"] = (
    category_index
    .groupby("category")["category_index"]
    .transform(lambda x: x.ffill().bfill().fillna(x.mean()))
)

# Floor at 15% of category mean — Embu always has some baseline demand
floor = category_index.groupby("category")["category_index"].transform(
    lambda x: x.mean() * 0.15
)
category_index["category_index"] = (
    category_index["category_index"].clip(lower=floor).round(4)
)

# ── Save ──────────────────────────────────────────────────────────────────────
raw_path = f"{OUTPUT_DIR}/embu_trends_monthly_{TODAY}.csv"
idx_path = f"{OUTPUT_DIR}/embu_trends_category_index_{TODAY}.csv"

raw_df.to_csv(raw_path, index=False)
category_index.to_csv(idx_path, index=False)

print("\nCategory index summary (0–1 scale):")
print(category_index.groupby("category")["category_index"]
      .describe().round(3).to_string())
print(f"\n✓ {raw_path}  ({len(raw_df):,} rows)")
print(f"✓ {idx_path}  ({len(category_index)} rows)")
print("\nRun dispensing_simulation.py next — it reads the index automatically.")