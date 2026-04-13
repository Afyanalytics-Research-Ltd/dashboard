"""
Embu Google Trends Pull — Simulation Signal (SerpAPI)
======================================================
Pulls monthly Google Trends data for Kenya (geo="KE") via SerpAPI.
No rate limiting, no 429s, no VPNs needed.

Run this script first, then run dispensing_simulation.py.

Requirements:
  pip install serpapi pandas python-dotenv

Outputs:
  embu_trends/
  ├── checkpoints/                         ← one CSV per subgroup
  ├── embu_trends_monthly_DATE.csv
  └── embu_trends_category_index_DATE.csv  ← consumed by simulation
"""

import os
import serpapi
import warnings
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

warnings.filterwarnings("ignore")
load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
TODAY          = datetime.now().strftime("%Y-%m-%d")
TIMEFRAME      = "2023-01-01 " + TODAY
GEO            = "KE"
SERPAPI_KEY    = os.getenv("SERPAPI_KEY")
OUTPUT_DIR     = "embu_trends"
CHECKPOINT_DIR = "embu_trends/checkpoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

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

def parse_timeline(results):
    """
    Parse SerpAPI interest_over_time response into a long-format DataFrame.
    Handles string/integer values, missing keys, and all-zero columns.
    """
    if "interest_over_time" not in results:
        return pd.DataFrame()

    timeline = results["interest_over_time"].get("timeline_data", [])
    if not timeline:
        return pd.DataFrame()

    def _clean_date(s):
        """Convert 'Jan 1 – 7, 2023' → '2023-01-01'. Falls back gracefully."""
        parts = str(s).split("–")
        if len(parts) == 2:
            year = parts[1].strip().split(",")[-1].strip()
            s = parts[0].strip() + ", " + year
        try:
            return pd.to_datetime(s, format="mixed").strftime("%Y-%m-%d")
        except Exception:
            return s

    rows = []
    for point in timeline:
        month = _clean_date(point.get("date") or point.get("timestamp", ""))
        for val in point.get("values", []):
            kw  = val.get("query", val.get("keyword", ""))
            raw = val.get("value", 0)
            try:
                score = 0 if str(raw) in ("<1", "", "None") else int(raw)
            except (ValueError, TypeError):
                score = 0
            rows.append({
                "month"          : month,
                "keyword"        : kw,
                "interest_score" : score,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Drop keywords that are all zeros — no signal
    nonzero = df.groupby("keyword")["interest_score"].sum()
    valid   = nonzero[nonzero > 0].index
    return df[df["keyword"].isin(valid)]


def fetch_monthly(keywords):
    """
    Pull monthly interest_over_time via SerpAPI Google Trends engine.
    Uses serpapi.GoogleSearch — the google-search-results package syntax.
    Returns long-format DataFrame or empty DataFrame.
    """
    try:
        results = serpapi.GoogleSearch({
            "engine"   : "google_trends",
            "q"        : ", ".join(keywords),
            "geo"      : GEO,
            "date"     : TIMEFRAME,
            "data_type": "TIMESERIES",
            "api_key"  : SERPAPI_KEY,
        }).get_dict()

        return parse_timeline(results)

    except Exception as e:
        print(f"\n  ✗  SerpAPI error: {e} — skipping")
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
remaining = total_subs - done_count

print("=" * 65)
print(f"  Embu Google Trends Pull  |  SerpAPI  |  geo={GEO}")
print(f"  Period   : {TIMEFRAME}")
print(f"  Progress : {done_count}/{total_subs} done  |  {remaining} remaining")
print(f"  Est. time: ~{remaining * 2} seconds")
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
        df = fetch_monthly(keywords)

        if df.empty:
            skipped.append(f"{category} / {sub_name}")
            print("✗  no data returned")
            continue

        # SerpAPI returns long format directly — just add labels
        df["category"] = category
        df["subgroup"] = sub_name

        save_checkpoint(df, category, sub_name)
        completed.append(f"{category} / {sub_name}")
        print(f"✓  {df['month'].nunique()} months — checkpoint saved")

print(f"\n{'='*65}")
print(f"Session complete — {len(completed)} done, {len(skipped)} skipped")
if skipped:
    print("Skipped subgroups:")
    for s in skipped:
        print(f"  - {s}")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD CATEGORY INDEX FROM ALL CHECKPOINTS
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding category index from all checkpoints...")
raw_df = load_all_checkpoints()

if raw_df.empty:
    raise RuntimeError(
        "No checkpoint data found. All subgroups returned empty. "
        "Check your SERPAPI_KEY in .env and re-run."
    )

def _clean_month(s):
    parts = str(s).split("–")
    if len(parts) == 2:
        year = parts[1].strip().split(",")[-1].strip()
        s = parts[0].strip() + ", " + year
    return pd.to_datetime(s, format="mixed")

raw_df["month"] = raw_df["month"].apply(_clean_month)

# Normalise each keyword to 0-1 across its full history
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

# Fill any missing months
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

# Floor at 15% of category mean
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