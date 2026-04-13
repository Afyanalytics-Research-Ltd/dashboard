"""
Google Trends External Signal Pipeline — Tendri KNN Model Integration
======================================================================
Where it slots in your notebook:
  After  → embu_external is defined (end of Step 1 / Get External Data)
  Before → Step 4 (Build the Embu target profile)

Outputs:
  google_trends_output/
  ├── checkpoints/                          ← one CSV per subgroup
  ├── beauty_skincare_weekly_DATE.csv
  ├── vitamins_supplements_weekly_DATE.csv
  ├── body_building_weekly_DATE.csv
  ├── master_weekly_wide_DATE.csv
  ├── master_weekly_long_DATE.csv
  ├── master_related_queries_DATE.csv
  └── embu_google_trends_features_DATE.csv  ← feeds into embu_external

Requirements:
    pip install google-search-results pandas python-dotenv
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
TIMEFRAME      = "today 12-m"
GEO            = "KE"
SERPAPI_KEY    = os.getenv("SERPAPI_KEY")
OUTPUT_DIR     = "google_trends_output"
CHECKPOINT_DIR = "google_trends_output/checkpoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def checkpoint_path(group_name, sub_name):
    return f"{CHECKPOINT_DIR}/{group_name}__{sub_name}.csv"

def already_done(group_name, sub_name):
    return os.path.exists(checkpoint_path(group_name, sub_name))

def save_checkpoint(df, group_name, sub_name):
    df.to_csv(checkpoint_path(group_name, sub_name), index=False)

def load_all_checkpoints():
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

def fetch_interest_over_time(keywords):
    """
    Pull weekly interest_over_time via SerpAPI Google Trends engine.
    Returns wide-format DataFrame (one column per keyword) or empty DataFrame.
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

        if "interest_over_time" not in results:
            return pd.DataFrame()

        records = {}
        for point in results["interest_over_time"]["timeline_data"]:
            date = point["date"]
            for val in point["values"]:
                kw    = val["query"]
                score = 0 if val["value"] == "<1" else int(val["value"])
                if kw not in records:
                    records[kw] = {}
                records[kw][date] = score

        df = pd.DataFrame(records)
        df.index.name = "date"
        return df

    except Exception as e:
        print(f"\n  ✗  SerpAPI error: {e} — skipping")
        return pd.DataFrame()


def fetch_related_queries(keywords):
    """
    Pull rising + top related queries via SerpAPI.
    Returns flat DataFrame: keyword | query_type | query | value
    """
    try:
        results = serpapi.GoogleSearch({
            "engine"   : "google_trends",
            "q"        : ", ".join(keywords),
            "geo"      : GEO,
            "date"     : TIMEFRAME,
            "data_type": "RELATED_QUERIES",
            "api_key"  : SERPAPI_KEY,
        }).get_dict()

        rows = []
        for kw, data in results.get("related_queries", {}).items():
            for qtype in ["top", "rising"]:
                for item in data.get(qtype, []):
                    rows.append({
                        "keyword"   : kw,
                        "query_type": qtype,
                        "query"     : item.get("query", ""),
                        "value"     : item.get("value", 0),
                    })
        return pd.DataFrame(rows)

    except Exception as e:
        print(f"\n  ✗  SerpAPI related queries error: {e} — skipping")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# Collapses each weekly series into 6 scalar features (all 0-1 scaled)
# ══════════════════════════════════════════════════════════════════════════════

def compute_trend_features(df_time):
    """
    Features per keyword:
      gt_{kw}_avg       — mean interest over 12 months        (0-1)
      gt_{kw}_peak      — max interest week                   (0-1)
      gt_{kw}_recent4w  — last 4 weeks average                (0-1)
      gt_{kw}_slope     — linear trend direction (+/- 1)
      gt_{kw}_spikes    — fraction of weeks with score > 75   (0-1)
      gt_{kw}_momentum  — recent4w / avg, capped + scaled     (0-1)
    """
    features = {}
    for col in df_time.columns:
        series = df_time[col].dropna()
        if series.empty:
            continue

        avg      = series.mean()
        peak     = series.max()
        recent4w = series.iloc[-4:].mean() if len(series) >= 4 else avg
        slope    = pd.Series(range(len(series))).corr(series)
        spikes   = (series > 75).sum() / max(len(series), 1)
        momentum = min((recent4w / avg) if avg > 0 else 1.0, 2.0) / 2.0

        safe = col.strip().lower().replace(" ","_").replace("/","_").replace("-","_")

        features[f"gt_{safe}_avg"]      = round(avg / 100, 4)
        features[f"gt_{safe}_peak"]     = round(peak / 100, 4)
        features[f"gt_{safe}_recent4w"] = round(recent4w / 100, 4)
        features[f"gt_{safe}_slope"]    = round(float(slope), 4) if not pd.isna(slope) else 0.0
        features[f"gt_{safe}_spikes"]   = round(float(spikes), 4)
        features[f"gt_{safe}_momentum"] = round(float(momentum), 4)

    return features


# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD GROUPS
# ══════════════════════════════════════════════════════════════════════════════

KEYWORD_GROUPS = {

    "beauty_skincare": {
        "subgroups": {
            "skincare_brands_1":          ["Nivea", "Neutrogena", "CeraVe", "COSRX", "La Roche-Posay"],
            "skincare_brands_2":          ["EOS", "Olay", "The Ordinary", "Bioderma", "Vaseline"],
            "skincare_products_1":        ["face serum", "moisturiser", "toner", "body lotion", "face wash"],
            "skincare_products_2":        ["vitamin C serum", "retinol cream", "hyaluronic acid", "niacinamide serum", "exfoliator"],
            "haircare_brands_1":          ["Dark and Lovely", "Cantu", "ORS", "Sunsilk", "Dove shampoo"],
            "haircare_brands_2":          ["Schwarzkopf", "Pantene", "TRESemmé", "Head and Shoulders", "Garnier hair"],
            "haircare_products_1":        ["hair relaxer", "hair growth oil", "deep conditioner", "natural hair products", "anti dandruff shampoo"],
            "haircare_products_2":        ["hair mask", "leave in conditioner", "edge control", "heat protectant", "hair serum"],
            "beauty_cosmetics_brands_1":  ["Maybelline", "Revlon", "Black Opal", "Wet n Wild", "Ruby Rose"],
            "beauty_cosmetics_brands_2":  ["e.l.f cosmetics", "Milani", "LA Girl", "Catrice", "essence cosmetics"],
            "makeup_brands_1":            ["NYX cosmetics", "MAC cosmetics", "Sleek makeup", "Rimmel", "Flormar"],
            "makeup_brands_2":            ["Charlotte Tilbury", "Urban Decay", "Too Faced", "NARS", "Fenty Beauty"],
            "makeup_products_1":          ["foundation", "lipstick", "mascara", "setting powder", "concealer"],
            "makeup_products_2":          ["eyeshadow palette", "blush", "highlighter makeup", "lip gloss", "eyeliner"],
            "sunscreen_brands_1":         ["Neutrogena sunscreen", "Nivea sun", "La Roche-Posay", "Bioderma sunscreen", "Garnier sunscreen"],
            "sunscreen_brands_2":         ["Eucerin sun", "Coppertone", "Ambre Solaire", "ISDIN", "Altruist sunscreen"],
            "sunscreen_products_1":       ["sunscreen", "SPF 50", "tinted sunscreen", "daily sunscreen", "sunblock"],
            "sunscreen_products_2":       ["sunscreen for black skin", "mineral sunscreen", "face sunscreen", "body sunscreen", "UV protection cream"],
        }
    },

    "vitamins_supplements": {
        "subgroups": {
            "immune_brands_1":                  ["Redoxon", "Berocca", "Ester-C", "Zinnat", "Supavit"],
            "immune_brands_2":                  ["Emergen-C", "Nature's Bounty vitamin C", "Solgar vitamin C", "Blackmores vitamin C", "Holland and Barrett vitamin C"],
            "immune_products_1":                ["Vitamin C supplement", "zinc supplement", "effervescent vitamin C", "immune booster", "vitamin C 1000mg"],
            "immune_products_2":                ["vitamin C tablets", "zinc and vitamin C", "vitamin C powder", "immune support supplement", "elderberry supplement"],
            "multivitamin_brands_1":            ["Centrum", "Seven Seas", "Supradyn", "Vitabiotics", "Abidec"],
            "multivitamin_brands_2":            ["Blackmores", "Solgar multivitamin", "Nature Made", "Garden of Life", "Kirkland vitamins"],
            "multivitamin_products_1":          ["multivitamin", "prenatal vitamins", "Wellwoman", "Wellman", "children multivitamin"],
            "multivitamin_products_2":          ["mens multivitamin", "womens multivitamin", "senior multivitamin", "gummy vitamins", "daily vitamins"],
            "beauty_supplement_brands_1":       ["Perfectil", "Nourkrin", "NeoCell", "Nature's Bounty biotin", "Zeta White"],
            "beauty_supplement_brands_2":       ["Vital Proteins", "Sports Research collagen", "Further Food collagen", "Ancient Nutrition", "Garden of Life collagen"],
            "beauty_supplement_products_1":     ["collagen supplement", "biotin supplement", "marine collagen", "collagen powder", "hair skin nails supplement"],
            "beauty_supplement_products_2":     ["collagen tablets", "biotin 10000mcg", "collagen drink", "beauty vitamins", "keratin supplement"],
            "bone_heart_brands_1":              ["Seven Seas Cod Liver Oil", "Omega H3", "Calcichew", "Caltrate", "Cardiowell"],
            "bone_heart_brands_2":              ["Blackmores fish oil", "Solgar omega 3", "Nature Made fish oil", "Nordic Naturals", "Kirkland fish oil"],
            "bone_heart_products_1":            ["Vitamin D supplement", "calcium supplement", "omega 3 fish oil", "cod liver oil", "Calcium D3"],
            "bone_heart_products_2":            ["vitamin D3", "calcium magnesium zinc", "fish oil capsules", "heart health supplement", "bone supplement"],
            "energy_stress_brands_1":           ["Neurobion", "Becosules", "Slow-Mag", "Magnesium B6", "Berocca Performance"],
            "energy_stress_brands_2":           ["Solgar B complex", "Nature Made B12", "Blackmores B complex", "Metagenics magnesium", "Pure Encapsulations"],
            "energy_stress_products_1":         ["B complex vitamin", "magnesium supplement", "energy supplement", "stress relief supplement", "vitamin B12"],
            "energy_stress_products_2":         ["magnesium glycinate", "ashwagandha", "B12 injection", "adaptogen supplement", "fatigue supplement"],
        }
    },

    "body_building": {
        "subgroups": {
            "protein_brands_1":         ["Optimum Nutrition", "USN protein", "BSN Syntha-6", "Muscletech", "Evox protein"],
            "protein_brands_2":         ["Dymatize protein", "Isopure", "MyProtein", "MuscleMeds", "Rule 1 protein"],
            "protein_products_1":       ["whey protein", "protein powder", "isolate protein", "plant protein", "protein shake"],
            "protein_products_2":       ["casein protein", "egg white protein", "protein bar", "whey concentrate", "vegan protein powder"],
            "mass_gainer_brands_1":     ["Serious Mass", "USN Muscle Fuel", "Dymatize Super Mass", "Mutant Mass", "Evox Mass"],
            "mass_gainer_brands_2":     ["Optimum Nutrition mass", "BSN True Mass", "Muscletech mass gainer", "MyProtein mass gainer", "Naked Mass"],
            "mass_gainer_products_1":   ["mass gainer", "weight gainer", "bulk supplement", "mass gainer chocolate", "3000 calorie shake"],
            "mass_gainer_products_2":   ["high calorie supplement", "hardgainer supplement", "mass gainer vanilla", "lean mass gainer", "mass gainer price"],
            "creatine_brands_1":        ["Creapure", "Optimum Nutrition creatine", "USN creatine", "Muscletech creatine", "BPI creatine"],
            "creatine_brands_2":        ["Kaged creatine", "Klean Athlete creatine", "Bulk Powders creatine", "MyProtein creatine", "Allmax creatine"],
            "creatine_products_1":      ["creatine", "creatine monohydrate", "creatine powder", "creatine supplement", "gym creatine"],
            "creatine_products_2":      ["creatine HCL", "creatine loading", "creatine capsules", "creatine and protein", "creatine for women"],
            "preworkout_brands_1":      ["C4 pre workout", "NO Xplode", "Ghost pre workout", "USN 3XT", "Muscletech Nano X"],
            "preworkout_brands_2":      ["Kaged Pre-Kaged", "Gorilla Mode", "Total War pre workout", "Wrecked pre workout", "Bucked Up"],
            "preworkout_products_1":    ["pre workout", "energy booster gym", "pre workout powder", "caffeine pre workout", "beta alanine"],
            "preworkout_products_2":    ["stim free pre workout", "pre workout drink", "pump supplement", "nitric oxide supplement", "citrulline supplement"],
            "amino_brands_1":           ["Scivation Xtend", "Optimum Nutrition BCAA", "USN BCAA", "Evox BCAA", "Muscletech amino"],
            "amino_brands_2":           ["Kaged BCAA", "Cellucor BCAA", "MyProtein BCAA", "BSN amino", "Allmax BCAA"],
            "amino_products_1":         ["BCAA", "EAA supplement", "amino acids", "glutamine supplement", "post workout supplement"],
            "amino_products_2":         ["BCAA powder", "recovery supplement", "BCAA capsules", "intra workout supplement", "electrolyte supplement"],
        }
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

all_time_series    = []
all_related        = []
embu_google_trends = {}

total_subs = sum(len(cfg["subgroups"]) for cfg in KEYWORD_GROUPS.values())
done_count = sum(
    already_done(g, s)
    for g, cfg in KEYWORD_GROUPS.items()
    for s in cfg["subgroups"]
)
remaining = total_subs - done_count

print("=" * 65)
print(f"  Google Trends → Tendri KNN  |  SerpAPI  |  {GEO}")
print(f"  Period   : {TIMEFRAME}")
print(f"  Progress : {done_count}/{total_subs} done  |  {remaining} remaining")
print(f"  Est. time: ~{remaining * 2} seconds")
print("=" * 65)

seq = 0

for group_name, group_cfg in KEYWORD_GROUPS.items():
    subgroups = group_cfg["subgroups"]
    print(f"\n[{group_name.upper()}]")
    group_series = []

    for sub_name, keywords in subgroups.items():
        seq += 1
        label = f"[{seq}/{total_subs}] {sub_name}"

        if already_done(group_name, sub_name):
            print(f"  {label} — ✓ already done")
            ckpt = pd.read_csv(checkpoint_path(group_name, sub_name))
            if "score" in ckpt.columns and "keyword" in ckpt.columns:
                wide = ckpt.pivot(index="date", columns="keyword", values="score")
                embu_google_trends.update(compute_trend_features(wide))
            continue

        print(f"  {label}...", end=" ", flush=True)

        df_time = fetch_interest_over_time(keywords)
        if not df_time.empty:
            embu_google_trends.update(compute_trend_features(df_time))
            df_ckpt = df_time.reset_index().melt(
                id_vars="date", var_name="keyword", value_name="score"
            )
            df_ckpt["group"]    = group_name
            df_ckpt["subgroup"] = sub_name
            save_checkpoint(df_ckpt, group_name, sub_name)
            df_wide = df_time.reset_index()
            df_wide["group"]    = group_name
            df_wide["subgroup"] = sub_name
            group_series.append(df_wide)
            all_time_series.append(df_wide)
            print(f"✓  {len(df_time)} weeks — checkpoint saved")
        else:
            print("✗  no data")

        df_rel = fetch_related_queries(keywords)
        if not df_rel.empty:
            df_rel["group"]    = group_name
            df_rel["subgroup"] = sub_name
            all_related.append(df_rel)

    # Per-group CSV
    if group_series:
        pd.concat(group_series, ignore_index=True).to_csv(
            f"{OUTPUT_DIR}/{group_name}_weekly_{TODAY}.csv", index=False
        )
        print(f"  → Saved: {group_name}_weekly_{TODAY}.csv")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE MASTER OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)

if all_time_series:
    master_weekly = pd.concat(all_time_series, ignore_index=True)

    # Wide format
    master_weekly.to_csv(
        f"{OUTPUT_DIR}/master_weekly_wide_{TODAY}.csv", index=False
    )
    print(f"  ✓ master_weekly_wide_{TODAY}.csv")

    # Long format — one row per keyword per week
    meta_cols    = ["date", "group", "subgroup"]
    keyword_cols = [c for c in master_weekly.columns if c not in meta_cols]
    master_long  = master_weekly.melt(
        id_vars    = meta_cols,
        value_vars = keyword_cols,
        var_name   = "keyword",
        value_name = "score",
    ).sort_values(["group", "subgroup", "keyword", "date"]).reset_index(drop=True)

    master_long.to_csv(
        f"{OUTPUT_DIR}/master_weekly_long_{TODAY}.csv", index=False
    )
    print(f"  ✓ master_weekly_long_{TODAY}.csv  "
          f"({master_long['keyword'].nunique()} keywords × "
          f"{master_long['date'].nunique()} weeks = "
          f"{len(master_long):,} rows)")

if all_related:
    pd.concat(all_related, ignore_index=True).to_csv(
        f"{OUTPUT_DIR}/master_related_queries_{TODAY}.csv", index=False
    )
    print(f"  ✓ master_related_queries_{TODAY}.csv")

# Scalar features for KNN model
gt_df = pd.DataFrame(
    list(embu_google_trends.items()), columns=["feature", "value"]
)
gt_df.to_csv(f"{OUTPUT_DIR}/embu_google_trends_features_{TODAY}.csv", index=False)
print(f"  ✓ embu_google_trends_features_{TODAY}.csv  ({len(gt_df)} model features)")
print(f"\n  Total features generated: {len(embu_google_trends)}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL INTEGRATION BLOCK
# Paste into your notebook after embu_external is defined, before Step 4
# ══════════════════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║  PASTE THIS BLOCK INTO YOUR NOTEBOOK                         ║
║  Location: after embu_external dict, before Step 4           ║
╚══════════════════════════════════════════════════════════════╝

import glob, pandas as pd

_path = sorted(glob.glob("google_trends_output/embu_google_trends_features_*.csv"))[-1]
embu_google_trends = dict(zip(*pd.read_csv(_path).values.T))

embu_external.update(embu_google_trends)

for feat in embu_google_trends:
    facility_profiles[feat] = 0

print(f"✓ {len(embu_google_trends)} Google Trends features added to embu_external")
""")