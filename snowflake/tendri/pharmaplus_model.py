"""
pharmaplus_model.py
===================
New Venture Headstart — Branch #106 Embu
Predictive opening stock model (KNN + Bayesian CI + GT weighting)

Usage:
    python pharmaplus_model.py

Outputs:
    data_export.pkl
    branch_106_opening_stock_products.csv

Fixes applied vs notebook:
  1. prod_names pulls from product_intel so new-category products are found
  2. Single clean pickle.dump including product_intel
  3. embu_trends_category_index always defined before Step 9
  4. No duplicate external_multiplier / compute_credible_intervals_gt definitions
"""

import pandas as pd
import numpy as np
import re
import glob
import os
import json
import pickle
from math import radians, sin, cos, sqrt, atan2
from collections import Counter
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

import fitz  # pip install pymupdf

from sqlalchemy import create_engine
import warnings
warnings.filterwarnings("ignore")

print("All imports loaded successfully")


# ── 0. DATABASE CONNECTION ─────────────────────────────────────────────────────
DB_USER = os.getenv("DB_USER",  "root")
DB_PASS = os.getenv("DB_PASS",  "ie97#")
DB_HOST = os.getenv("DB_HOST",  "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))

DATABASE  = os.getenv("DB_NAME", "tenri_raw")
TENRI     = os.getenv("DB_NAME", "tenri")
REPORTING = os.getenv("DB_NAME", "reporting")

tenri_raw_engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DATABASE}")
tenri            = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{TENRI}")
reporting        = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{REPORTING}")

monthly_diagnoses_aggregate  = pd.read_sql_query("SELECT * FROM tenri_raw.monthly_diagnoses_aggregate",  tenri_raw_engine)
monthly_dispensing_aggregate = pd.read_sql_query("SELECT * FROM tenri_raw.monthly_dispensing_aggregate", tenri_raw_engine)
print("Real data loaded")


# ── CELL A: Append Simulated Dispensing Aggregate ──────────────────────────────
_sim_agg_path = sorted(glob.glob(
    r"C:\Users\Mercy\Documents\Tendri\Snowflake Pulls\Xana\snowflake\tendri\data\simulated_dispensing_aggregate_*.csv"
))

if not _sim_agg_path:
    print("WARNING: No simulated_dispensing_aggregate_*.csv found. Run dispensing_simulation.py first.")
else:
    sim_agg = pd.read_csv(_sim_agg_path[-1])

    REQUIRED_COLS = [
        "facility_id", "months", "product_id", "new_category_name",
        "parent_category_name", "correct_therapeutic_class",
        "total_qty_dispensed", "unique_patients", "avg_qty_per_patient"
    ]
    missing = [c for c in REQUIRED_COLS if c not in sim_agg.columns]
    if missing:
        raise ValueError(f"Simulated data missing columns: {missing}")

    real_pids = set(monthly_dispensing_aggregate["product_id"].unique())
    sim_pids  = set(sim_agg["product_id"].unique())
    clash     = real_pids & sim_pids
    if clash:
        raise ValueError(f"product_id collision: {len(clash)} IDs. Sample: {list(clash)[:5]}")

    sim_agg["facility_id"]         = sim_agg["facility_id"].astype("int64")
    sim_agg["product_id"]          = sim_agg["product_id"].astype("int64")
    sim_agg["total_qty_dispensed"] = sim_agg["total_qty_dispensed"].astype("float64")
    sim_agg["unique_patients"]     = sim_agg["unique_patients"].astype("int64")
    sim_agg["avg_qty_per_patient"] = sim_agg["avg_qty_per_patient"].astype("float64")
    sim_agg["months"]              = pd.to_datetime(sim_agg["months"]).dt.strftime("%Y-%m-%d")

    monthly_dispensing_aggregate = pd.concat(
        [monthly_dispensing_aggregate[REQUIRED_COLS], sim_agg[REQUIRED_COLS]],
        ignore_index=True
    )

    print(f"monthly_dispensing_aggregate updated: {len(monthly_dispensing_aggregate):,} rows | "
          f"{monthly_dispensing_aggregate['new_category_name'].nunique()} categories")
    for cat, n in monthly_dispensing_aggregate["new_category_name"].value_counts().items():
        tag = " ← NEW" if cat in ["Beauty Products", "Vitamins & Supplements", "Body Building"] else ""
        print(f"    {cat:<35} {n:>8,} rows{tag}")


# ── Fact tables ────────────────────────────────────────────────────────────────
fact_dispensing         = pd.read_sql_query("SELECT * FROM tenri.fact_dispensing",          tenri)
fact_inventory_snapshot = pd.read_sql_query("SELECT * FROM tenri.fact_inventory_snapshot",  tenri)
dim_patient_profile     = pd.read_sql_query("SELECT * FROM tenri.dim_patient_profile",      tenri)
settings_facility       = pd.read_sql_query("SELECT * FROM tenri.settings_clinics",         tenri)
print("Fact tables loaded")


# ── CELL B: Append Simulated Fact Tables ──────────────────────────────────────
_sim_disp_path  = sorted(glob.glob(r"C:\Users\Mercy\Documents\Tendri\Snowflake Pulls\Xana\snowflake\tendri\data\simulated_fact_dispensing_*.csv"))
_sim_snap_path  = sorted(glob.glob(r"C:\Users\Mercy\Documents\Tendri\Snowflake Pulls\Xana\snowflake\tendri\data\simulated_inventory_snapshot_*.csv"))
_sim_intel_path = sorted(glob.glob(r"C:\Users\Mercy\Documents\Tendri\Snowflake Pulls\Xana\snowflake\tendri\data\simulated_product_intelligence_*.csv"))

if _sim_disp_path:
    sim_disp = pd.read_csv(_sim_disp_path[-1], parse_dates=["date"])
    FACT_DISP_COLS = [c for c in fact_dispensing.columns if c in sim_disp.columns]
    sim_disp_clean = sim_disp[FACT_DISP_COLS].copy()
    sim_disp_clean["facility_id"] = sim_disp_clean["facility_id"].astype("int64")
    sim_disp_clean["product_id"]  = sim_disp_clean["product_id"].astype("int64")
    fact_dispensing = pd.concat([fact_dispensing, sim_disp_clean], ignore_index=True)
    print(f"fact_dispensing updated: {len(fact_dispensing):,} rows")
else:
    print("WARNING: No simulated_fact_dispensing_*.csv found — skipping.")

if _sim_snap_path:
    sim_snap = pd.read_csv(_sim_snap_path[-1], parse_dates=["snapshot_date"])
    SNAP_COLS = [c for c in fact_inventory_snapshot.columns if c in sim_snap.columns]
    sim_snap_clean = sim_snap[SNAP_COLS].copy()
    sim_snap_clean["facility_id"] = sim_snap_clean["facility_id"].astype("int64")
    sim_snap_clean["product_id"]  = sim_snap_clean["product_id"].astype("int64")
    fact_inventory_snapshot = pd.concat([fact_inventory_snapshot, sim_snap_clean], ignore_index=True)
    print(f"fact_inventory_snapshot updated: {len(fact_inventory_snapshot):,} rows")
else:
    print("WARNING: No simulated_inventory_snapshot_*.csv found — skipping.")

if _sim_intel_path:
    product_intel = pd.read_csv(_sim_intel_path[-1])
    print(f"product_intel loaded: {len(product_intel):,} products")
else:
    product_intel = pd.DataFrame()
    print("WARNING: No simulated_product_intelligence_*.csv found.")


# ── CONFIGURATION ──────────────────────────────────────────────────────────────
RAMP_UP_MONTHS         = 3
KNN_K                  = 3
KNN_FALLBACK_THRESHOLD = 0.05
DEAD_STOCK_MULTIPLIER  = 1.5
CONFIDENCE_LEVEL       = 0.90
print("Configuration set")


# ── CELL C: Rebuild disp_df with new categories ────────────────────────────────
diag_df = monthly_diagnoses_aggregate.copy()
disp_df = monthly_dispensing_aggregate.copy()
disp_df["months"] = pd.to_datetime(disp_df["months"])

diag_df["facility_id"] = diag_df["facility_id"].astype(int)
diag_df["monthly"]     = pd.to_datetime(diag_df["monthly"])

print(f"disp_df rebuilt: {len(disp_df):,} rows | {disp_df['new_category_name'].nunique()} categories")
for cat in ["Beauty Products", "Vitamins & Supplements", "Body Building"]:
    n = (disp_df["new_category_name"] == cat).sum()
    print(f"  {'✓' if n > 0 else '✗ MISSING'} {cat}: {n:,} rows")


# ── GEOJSON + PROXIMITY ────────────────────────────────────────────────────────
geojson_path = r"C:/Users/Mercy/Documents/Tendri/Snowflake Pulls/Xana/snowflake/tendri/data/hotosm_ken_health_facilities_points_geojson.geojson"

with open(geojson_path, "r", encoding="utf-8") as f:
    geojson = json.load(f)

rows = []
for feature in geojson["features"]:
    row = feature["properties"].copy()
    coords = feature["geometry"]["coordinates"] if feature["geometry"] else [None, None]
    row["longitude"] = coords[0]
    row["latitude"]  = coords[1]
    rows.append(row)

kenya_health_facilities = pd.DataFrame(rows)

diag_df["facility_id"] = diag_df["facility_key"].str.split("|").str[-1].astype(int)

maternity_signal = (
    diag_df.groupby("facility_id")
    .agg(pregnant_cases=("pregnant_cases", "sum"),
         total_consultations=("consultation_count", "sum"),
         total_female=("total_female", "sum"),
         total_male=("total_male", "sum"))
    .reset_index()
)
maternity_signal["pregnant_rate"] = (
    maternity_signal["pregnant_cases"] / maternity_signal["total_consultations"] * 100
).round(2)

tenri_facilities = kenya_health_facilities[
    kenya_health_facilities["name"].str.contains("Tenri", case=False, na=False)
].copy()

maternity_facility_id = maternity_signal.sort_values("pregnant_cases", ascending=False).iloc[0]["facility_id"]
general_facility_id   = maternity_signal.sort_values("pregnant_cases", ascending=False).iloc[1]["facility_id"]
maternity_coords      = tenri_facilities[tenri_facilities["name"].str.contains("Mater", case=False, na=False)].iloc[0]
general_coords        = tenri_facilities[~tenri_facilities["name"].str.contains("Mater", case=False, na=False)].iloc[0]

facility_coords = pd.DataFrame([
    {"facility_id": int(maternity_facility_id), "name": maternity_coords["name"],
     "latitude": maternity_coords["latitude"], "longitude": maternity_coords["longitude"], "type": "Maternity & Theatre"},
    {"facility_id": int(general_facility_id), "name": general_coords["name"],
     "latitude": general_coords["latitude"], "longitude": general_coords["longitude"], "type": "General Hospital"},
])


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


RADIUS_KM      = 25
nearby_results = []

for _, tenri_fac in facility_coords.iterrows():
    lat1, lon1 = tenri_fac["latitude"], tenri_fac["longitude"]
    kdf = kenya_health_facilities.copy().dropna(subset=["latitude", "longitude"])
    kdf = kdf[~kdf["name"].str.contains("Tenri", case=False, na=False)]
    kdf["distance_km"] = kdf.apply(lambda row: haversine_km(lat1, lon1, row["latitude"], row["longitude"]), axis=1)
    nearby = kdf[kdf["distance_km"] <= RADIUS_KM].sort_values("distance_km").copy()
    nearby["tenri_facility_id"]   = tenri_fac["facility_id"]
    nearby["tenri_facility_name"] = tenri_fac["name"]
    nearby_results.append(nearby)

nearby_all = pd.concat(nearby_results, ignore_index=True)

proximity_features = []
for facility_id in facility_coords["facility_id"]:
    nearby = nearby_all[nearby_all["tenri_facility_id"] == facility_id]
    proximity_features.append({
        "facility_id":           facility_id,
        "n_facilities_25km":     len(nearby),
        "n_hospitals_25km":      (nearby["amenity"] == "hospital").sum(),
        "n_clinics_25km":        (nearby["amenity"].isin(["clinic", "doctors"])).sum(),
        "n_pharmacies_25km":     (nearby["amenity"] == "pharmacy").sum(),
        "n_health_centres_25km": (nearby["healthcare"].str.contains("health_centre|health centre", case=False, na=False)).sum(),
        "has_hospital_nearby":   int((nearby["amenity"] == "hospital").any()),
    })

proximity_df = pd.DataFrame(proximity_features)

FACILITY_DEMAND_MAP = {
    "hospital":      ["Injectables", "IV Fluids & Infusions", "Wound Care", "Oral Solid Forms", "Infection Control"],
    "clinic":        ["Oral Solid Forms", "Oral Liquid Forms", "Injectables"],
    "doctors":       ["Oral Solid Forms", "Oral Liquid Forms"],
    "pharmacy":      ["Oral Solid Forms", "Topical Preparations"],
    "maternity":     ["Injectables", "Oral Solid Forms", "Vaccines & Biologicals", "IV Fluids & Infusions"],
    "health_centre": ["Oral Solid Forms", "Oral Liquid Forms", "Injectables", "Vaccines & Biologicals"],
}

category_proximity_driver = {}
for _, fac in nearby_all.iterrows():
    amenity    = str(fac.get("amenity",    "")).lower()
    healthcare = str(fac.get("healthcare", "")).lower()
    name       = str(fac.get("name", "Unknown facility"))
    dist       = round(fac.get("distance_km", 0), 1)
    if "maternity" in name.lower():                     fac_type = "maternity"
    elif amenity in FACILITY_DEMAND_MAP:                fac_type = amenity
    elif "health_centre" in healthcare or "health centre" in healthcare: fac_type = "health_centre"
    else: continue
    for category in FACILITY_DEMAND_MAP[fac_type]:
        if category not in category_proximity_driver: category_proximity_driver[category] = []
        entry = f"{name} ({dist}km)"
        if entry not in category_proximity_driver[category]: category_proximity_driver[category].append(entry)
for cat in category_proximity_driver:
    category_proximity_driver[cat] = category_proximity_driver[cat][:2]


# ── DHS EXTERNAL DATA ──────────────────────────────────────────────────────────
embu_dhs = pd.read_csv(r"C:\Users\Mercy\Documents\Tendri\Snowflake Pulls\Xana\snowflake\tendri\data\embu_dhs_2022.csv")
embu_dhs = embu_dhs.rename(columns={"Unnamed: 0": "geography"}).set_index("geography")


def safe_float(val):
    if pd.isna(val): return None
    s = str(val).replace("(", "").replace(")", "").replace("<", "").strip()
    try: return float(s)
    except: return None


embu_row = embu_dhs.loc["embu"]

embu_external = {
    "fertility_rate":             safe_float(embu_row["Total fertility rate (number of children per woman)"]),
    "teen_pregnancy_pct":         safe_float(embu_row["Teenage pregnancy (% age 15-19 who have ever been pregnant)"]),
    "modern_fp_use_pct":          safe_float(embu_row["Use of modern method of FP (% of married women age 15-49)"]),
    "antenatal_4plus_visits_pct": safe_float(embu_row["Women age 15-49 who had a live birth and had 4+ antenatal visits (%)"]),
    "skilled_birth_pct":          safe_float(embu_row["Births delivered by a skilled provider2 (%)"]),
    "u5_stunting_pct":            safe_float(embu_row["Children under 5 who are stunted (%) (too short for their age)"]),
    "u5_underweight_pct":         safe_float(embu_row["Children under 5 who are underweight (%) (too thin for their age)"]),
    "vaccination_pct":            safe_float(embu_row["Children age 12-23 months fully vaccinated (basic antigens)3 (%)"]),
    "itn_access_pct":             safe_float(embu_row["Household population with access to an insecticide-treated net (ITN) (%)"]),
    "itn_use_pct":                safe_float(embu_row["Household population who slept under an ITN the night before the survey (%)"]),
    "clean_fuel_access_pct":      safe_float(embu_row["Household population relying on clean fuels and technologies for cooking, space heating, & lighting (%)"]),
    "water_access_pct":           safe_float(embu_row["Household population with access to at least basic drinking water service (%)"]),
    "sanitation_access_pct":      safe_float(embu_row["Household population with at least basic sanitation service (%)"]),
    "women_no_education_pct":     safe_float(embu_row["Women age 15-49 with no formal education (%)"]),
    "is_urban_branch":            1,
}
print(f"embu_external: {len(embu_external)} DHS keys")


# ── KNBS PROJECTIONS + CIDP AGE COHORTS ───────────────────────────────────────
_PROJ_PDF = r"C:\Users\Mercy\Documents\Tendri\Snowflake Pulls\Xana\snowflake\tendri\data\2019-Kenya-population-and-Housing-Census-Summary-Report-on-Kenyas-Population-Projections.pdf"
_VOL1_PDF = r"C:\Users\Mercy\Documents\Tendri\Snowflake Pulls\Xana\snowflake\tendri\data\2019-Kenya-population-and-Housing-Census-Volume-1-Population-By-County-And-Sub-County.pdf"
_CIDP_PDF = r"C:\Users\Mercy\Documents\Tendri\Snowflake Pulls\Xana\snowflake\tendri\data\EMBU_CIDP_2023-2027.pdf"


def _pn(s):
    try: return int(str(s).replace(",", "").strip())
    except: return None


def _gl(doc, pg):
    return [l.strip() for l in doc[pg].get_text().split("\n") if l.strip()]


def _ct(doc, pg, n, skip):
    lines = _gl(doc, pg); rows, i = [], 0
    while i < len(lines):
        l = lines[i]
        if re.match(r"^[\d,]+$", l) or l in skip or len(l) < 2: i += 1; continue
        nums, j = [], i + 1
        while j < len(lines) and len(nums) < n:
            if re.match(r"^[\d,]+$", lines[j]): nums.append(_pn(lines[j])); j += 1
            else: break
        if len(nums) == n: rows.append({"county": l, "_nums": nums}); i = j
        else: i += 1
    return rows


_PY = ["2020","2021","2022","2023","2024","2025","2030","2035","2040","2045"]
_LY = ["2020","2021","2022","2023","2024","2025","2030","2035"]
_HY = ["2020","2021","2022","2023","2024","2025","2026","2027","2028","2029","2030"]

_dp  = fitz.open(_PROJ_PDF)
_pe  = next(r for r in _ct(_dp, 1, 10, {"County","Kenya"} | set(_PY) | {"Population Projections by County, 2020-2045"}) if r["county"] == "Embu")
_le  = next(r for r in _ct(_dp, 3, 8,  {"County","Kenya"} | set(_LY) | {"Population in the Labour Force, age 15-64 by County, 2020-2035","Labour Force Projections","The population in the labour force (age 15-64) is expected to increase by 40.7 percent from 28.8 million","in 2020 to 40.5 million by 2035."}) if r["county"] == "Embu")
_he  = next(r for r in _ct(_dp, 4, 11, {"County","Kenya"} | set(_HY) | {"Projected Number of Households by County, 2020-2030","Households Projections","Data on Household projections show that by 2030, there will be approximately 15.9 million households.","Nairobi City, which is entirely urban, will require nearly 2 million houses to host its population by 2030."}) if r["county"] == "Embu")
_proj = dict(zip(_PY, _pe["_nums"]))
_lf   = dict(zip(_LY, _le["_nums"]))
_hh   = dict(zip(_HY, _he["_nums"]))

_dv = fitz.open(_VOL1_PDF); _cr = []
for _pg in range(len(_dv)):
    _ls = [l.strip() for l in _dv[_pg].get_text().split("\n") if l.strip()]
    for _i, _l in enumerate(_ls):
        if not re.match(r"^[\d,]+$", _l) and len(_l) > 2 and _i + 4 < len(_ls):
            _c = []
            for _k in range(1, 5):
                if re.match(r"^[\d,]+$", _ls[_i + _k]): _c.append(_pn(_ls[_i + _k]))
                else: break
            if len(_c) == 4 and _c[3] == _c[0] + _c[1] + _c[2]:
                _cr.append({"county": _l, "male": _c[0], "female": _c[1], "total": _c[3]})
_ec = pd.DataFrame(_cr).drop_duplicates("county")
_ec = _ec[_ec["county"].str.contains("Embu", na=False)].iloc[0]

_T19 = int(_ec["total"]); _T25 = int(_proj["2025"])
_L25 = int(_lf["2025"]);  _H25 = int(_hh["2025"])
_AGR = (_T25 / _T19) ** (1 / 6) - 1

_dc = fitz.open(_CIDP_PDF); _CY = [2019, 2022, 2025, 2027]
_l5 = _gl(_dc, 44); _r5 = []; _i = 0
while _i < len(_l5):
    _ln = _l5[_i]
    if re.match(r"^\s*\d+[-+]\d*\s*$", _ln) or _ln in ("Age NS", "All Ages"):
        _ns, _j = [], _i + 1
        while _j < len(_l5) and len(_ns) < 12:
            _v = _l5[_j]
            if re.match(r"^[\d,]+$", _v): _ns.append(_pn(_v)); _j += 1
            elif _v == "-":  _ns.append(None); _j += 1
            else: break
        if len(_ns) == 12:
            for _yi, _yr in enumerate(_CY):
                _o = _yi * 3
                _r5.append({"age_cohort": _ln.strip(), "year": _yr, "male": _ns[_o], "female": _ns[_o+1], "total": _ns[_o+2]})
            _i = _j; continue
    _i += 1

_t5  = pd.DataFrame(_r5)
_s25 = _t5[_t5["year"] == 2025].set_index("age_cohort")


def _ct25(*g): return sum(_s25.loc[x, "total"] for x in g if x in _s25.index)
def _cm25(*g): return sum(_s25.loc[x, "male"]  for x in g if x in _s25.index)
def _cf25(*g): return sum(_s25.loc[x, "female"] for x in g if x in _s25.index)


embu_external.update({
    "embu_total_population_2025": _T25, "embu_labour_force_2025": _L25,
    "embu_households_2025": _H25,       "embu_annual_growth_rate": round(_AGR, 5),
    "embu_pop_2020": int(_proj["2020"]), "embu_pop_2021": int(_proj["2021"]),
    "embu_pop_2022": int(_proj["2022"]), "embu_pop_2023": int(_proj["2023"]),
    "embu_pop_2024": int(_proj["2024"]), "embu_pop_2025": int(_proj["2025"]),
    "embu_labour_force_pct": round(_L25 / _T25, 4),
    "embu_pct_female_2019":  round(float(_ec["female"]) / _T19, 4),
    "pct_under_15":     round(_ct25("0-4", "5-9", "10-14") / _T25, 4),
    "pct_youth_15_29":  round(_ct25("15-19", "20-24", "25-29") / _T25, 4),
    "pct_adult_30_64":  round(_ct25("30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64") / _T25, 4),
    "pct_aged_65_plus": round(_ct25("65-69", "70-74", "75-79", "80+") / _T25, 4),
    "beauty_target_pct":      round(_cf25("15-19","20-24","25-29","30-34","35-39","40-44","45-49") / _T25, 4),
    "bb_target_pct":          round(_cm25("15-19","20-24","25-29","30-34") / _T25, 4),
    "supplements_target_pct": round(_ct25("25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64") / _T25, 4),
})
print(f"embu_external: {len(embu_external)} total keys after KNBS + CIDP update")


# ── 2. DISEASE BURDEN MAPPING ──────────────────────────────────────────────────
BURDEN_MAP = {
    "malaria":         ["Communicable-Malaria", "Malaria"],
    "communicable":    ["Communicable-Infectious", "Communicable"],
    "hypertension":    ["NCD-Cardiovascular-Hypertension"],
    "cardiovascular":  ["NCD-Cardiovascular-CoronaryHeart","NCD-Cardiovascular-HeartFailure","NCD-Cardiovascular"],
    "diabetes":        ["NCD-Endocrine-Diabetes","NCD-Endocrine-Other","NCD-Endocrine"],
    "respiratory":     ["Respiratory-URTI","Respiratory-Pneumonia","Respiratory-Rhinitis-Sinusitis","NCD-Respiratory-Asthma","Respiratory"],
    "gi":              ["GI-Gastritis","GI-PepticUlcer","GI-GERD-Oesophageal"],
    "maternal":        ["MNCH-Maternal","MNCH"],
    "gynaecological":  ["GU-GynaeUrological"],
    "musculoskeletal": ["MSK-Musculoskeletal"],
    "dermatological":  ["Dermatological"],
    "mental_health":   ["NCD-Mental"],
}
print(f"Defined {len(BURDEN_MAP)} burden dimensions")


# ── 3. BUILD FACILITY PROFILES (Steps 3a–3e) ──────────────────────────────────
df = diag_df.copy()
burden_cols = [f"burden_{k}" for k in BURDEN_MAP]

for burden_name, keywords in BURDEN_MAP.items():
    segmented = [r"(?:^|\|)" + re.escape(kw) + r"(?=\||$)" for kw in keywords]
    pattern = "|".join(segmented)
    df[f"burden_{burden_name}"] = df["combined_diagnosis"].str.contains(pattern, case=False, na=False).astype(int)

df["age_paediatric"] = df["total_age_less_than_1"] + df["total_age_1_4"] + df["total_age_5_12"]
df["age_working"]    = df["total_age_18_24"] + df["total_age_25_34"] + df["total_age_35_44"]
df["age_chronic"]    = df["total_age_45_54"] + df["total_age_55_64"] + df["total_age_over_65"]

case_cols  = ["chronic_cases","pregnant_cases","follow_up_cases","immunisation_cases","medication_pickup_cases"]
count_cols = ["unique_patient_count","total_male","total_female","age_paediatric","age_working","age_chronic","consultation_count"]

facility_agg = df.groupby("facility_id")[burden_cols + case_cols + count_cols].sum().reset_index()

total_diagnoses = facility_agg[burden_cols].sum(axis=1).replace(0, np.nan)
total_consults  = facility_agg["consultation_count"].replace(0, np.nan)
total_patients  = facility_agg["unique_patient_count"].replace(0, np.nan)

for col in burden_cols: facility_agg[f"{col}_ratio"] = facility_agg[col] / total_diagnoses
for col in case_cols:   facility_agg[f"{col}_ratio"] = facility_agg[col] / total_consults
facility_agg["pct_paediatric"]  = facility_agg["age_paediatric"] / total_patients
facility_agg["pct_working"]     = facility_agg["age_working"]    / total_patients
facility_agg["pct_chronic_age"] = facility_agg["age_chronic"]    / total_patients
facility_agg["pct_female"]      = facility_agg["total_female"]   / total_patients

monthly_size = (
    diag_df.groupby(["facility_id","monthly"])["consultation_count"].sum()
    .groupby("facility_id").mean().rename("avg_monthly_consultations").reset_index()
)
facility_profiles = facility_agg.merge(monthly_size, on="facility_id", how="left")

disp_behaviour = (
    disp_df.groupby("facility_id")
    .agg(total_qty=("total_qty_dispensed","sum"),
         unique_products=("product_id","nunique"),
         avg_patients=("unique_patients","mean"))
    .reset_index()
)
cat_mix = disp_df.groupby(["facility_id","new_category_name"])["total_qty_dispensed"].sum().unstack(fill_value=0)
cat_mix = cat_mix.div(cat_mix.sum(axis=1), axis=0)
cat_mix.columns = [f"cat_ratio_{c.lower().replace(' ','_').replace('/','_')}" for c in cat_mix.columns]
cat_mix = cat_mix.reset_index()
cat_ratio_cols = [c for c in cat_mix.columns if c.startswith("cat_ratio_")]

facility_profiles = facility_profiles.merge(disp_behaviour, on="facility_id", how="left")
facility_profiles = facility_profiles.merge(cat_mix,         on="facility_id", how="left")
facility_profiles = facility_profiles.merge(proximity_df,    on="facility_id", how="left")

proximity_cols = ["n_facilities_25km","n_hospitals_25km","n_clinics_25km","n_pharmacies_25km","has_hospital_nearby"]
for col in proximity_cols: facility_profiles[col] = facility_profiles[col].fillna(0)

ratio_cols = (
    [f"{c}_ratio" for c in burden_cols + case_cols]
    + ["pct_paediatric","pct_working","pct_chronic_age","pct_female","avg_monthly_consultations"]
    + proximity_cols + cat_ratio_cols
)

missing = [c for c in ratio_cols if c not in facility_profiles.columns]
print(f"facility_profiles: {facility_profiles.shape} | ratio_cols: {len(ratio_cols)} features")
if missing: print(f"WARNING — missing: {missing}")
else: print("✓ All feature columns present")


# ── GT + PROXIMITY INTEGRATION ────────────────────────────────────────────────
GT_DIR   = r"C:\Users\Mercy\Documents\Tendri\Snowflake Pulls\Xana\snowflake\google_trends_output"
PROX_CSV = r"C:\Users\Mercy\Documents\Tendri\Snowflake Pulls\Xana\snowflake\tendri\data\pharmaplus_proximity_data.csv"

# PART 1: Google Trends
print("Loading Google Trends features...")
_gt_paths = sorted(glob.glob(os.path.join(GT_DIR, "embu_google_trends_features_*.csv")))

if not _gt_paths:
    print("  WARNING: No embu_google_trends_features_*.csv found — using empty dict.")
    embu_google_trends = {}
else:
    _gt_df = pd.read_csv(_gt_paths[-1])
    if not {"feature","value"}.issubset(_gt_df.columns):
        raise ValueError(f"Expected columns 'feature' and 'value', got: {_gt_df.columns.tolist()}")
    embu_google_trends = dict(zip(_gt_df["feature"], _gt_df["value"]))
    print(f"  Loaded {len(embu_google_trends)} features")

# Category index time series — always define even if file missing
_idx_paths = sorted(glob.glob(os.path.join(GT_DIR, "embu_trends_category_index_*.csv")))
if _idx_paths:
    embu_trends_category_index = pd.read_csv(_idx_paths[-1], parse_dates=["month"])
    print(f"  Category index loaded: {len(embu_trends_category_index)} rows")
else:
    embu_trends_category_index = pd.DataFrame()   # safe default — CI will use uniform weights
    print("  No category index found — GT-weighted CI will use uniform weights.")

# PART 2: Proximity
print("\nLoading proximity data...")
if not os.path.exists(PROX_CSV):
    print(f"  WARNING: {PROX_CSV} not found — using empty dict.")
    embu_proximity = {}
else:
    prox = pd.read_csv(PROX_CSV)
    prox["distance_km"] = pd.to_numeric(prox["distance_km"], errors="coerce")
    prox_dedup = prox.drop_duplicates(subset=["place_name","category"])
    prox_near  = prox_dedup[prox_dedup["distance_km"] <= 3.0].copy()

    CAT_MAP = {
        "gym_fitness": "gym", "beauty_salon_spa": "beauty_salon",
        "beauty_shop_cosmetics": "beauty_shop", "supplements_vitamins": "supplement_store",
        "bodybuilding_shop": "bb_shop", "pharmacy_competitor": "pharmacy", "supermarket": "supermarket",
    }
    NORMALISE_CAP = {"gym":10,"beauty_salon":20,"beauty_shop":15,"supplement_store":8,"bb_shop":5,"pharmacy":15,"supermarket":10}

    embu_proximity = {}
    for csv_cat, feat_prefix in CAT_MAP.items():
        subset      = prox_near[prox_near["category"] == csv_cat]
        cap         = NORMALISE_CAP[feat_prefix]
        count       = len(subset)
        chain_count = int(subset["is_chain"].sum()) if "is_chain" in subset.columns else 0
        ratings     = pd.to_numeric(subset["rating"], errors="coerce").dropna()
        distances   = subset["distance_km"].dropna()
        embu_proximity[f"n_{feat_prefix}_3km"]       = round(min(count / cap, 1.0), 4)
        embu_proximity[f"n_{feat_prefix}_chain_3km"] = round(min(chain_count / max(count, 1), 1.0), 4)
        embu_proximity[f"avg_{feat_prefix}_rating"]  = round(ratings.mean() / 5.0, 4) if len(ratings) > 0 else 0.0
        embu_proximity[f"nearest_{feat_prefix}_km"]  = round(1 - distances.min() / 3.0, 4) if len(distances) > 0 else 0.0

    embu_proximity["beauty_demand_proximity"]     = round((embu_proximity.get("n_beauty_salon_3km",0) + embu_proximity.get("n_beauty_shop_3km",0)) / 2, 4)
    embu_proximity["bb_demand_proximity"]         = round((embu_proximity.get("n_gym_3km",0) + embu_proximity.get("n_bb_shop_3km",0)) / 2, 4)
    embu_proximity["supplement_demand_proximity"] = round((embu_proximity.get("n_supplement_store_3km",0) + embu_proximity.get("n_gym_3km",0)) / 2, 4)
    embu_proximity["pharmacy_competition_index"]  = round((embu_proximity.get("n_pharmacy_3km",0) + embu_proximity.get("n_supermarket_3km",0)) / 2, 4)
    print(f"  Computed {len(embu_proximity)} proximity features")

# PART 3: Update embu_external
_before = len(embu_external)
embu_external.update(embu_google_trends)
embu_external.update(embu_proximity)
print(f"\nembu_external: {_before} → {len(embu_external)} keys (+{len(embu_external)-_before})")

# PART 4: Zero-fill facility_profiles
_new_feats = list(embu_google_trends.keys()) + list(embu_proximity.keys())
for feat in _new_feats:
    if feat not in facility_profiles.columns:
        facility_profiles[feat] = 0.0
print(f"facility_profiles zero-filled for {len(_new_feats)} new features")


# ── MODEL PIPELINE WIRING ──────────────────────────────────────────────────────
# Extend ratio_cols with GT + proximity keys (must happen before Step 4)
_new_external_features = list(embu_google_trends.keys()) + list(embu_proximity.keys())
_before = len(ratio_cols)
ratio_cols = list(dict.fromkeys(ratio_cols + _new_external_features))
print(f"ratio_cols: {_before} → {len(ratio_cols)} (+{len(ratio_cols)-_before})")

_missing_in_profiles = [f for f in _new_external_features if f not in facility_profiles.columns]
if _missing_in_profiles:
    for feat in _missing_in_profiles: facility_profiles[feat] = 0.0
    print(f"  Zero-filled {len(_missing_in_profiles)} missing features")
else:
    print("  ✓ All new features present in facility_profiles")


# GT-weighted Bayesian CI ──────────────────────────────────────────────────────
GT_CATEGORY_MAP = {
    "Beauty Products":        "beauty",
    "Vitamins & Supplements": "supplements",
    "Body Building":          "bodybuilding",
}


def compute_gt_weights(category_name, steady_df_subset):
    if embu_trends_category_index.empty:
        return np.ones(len(steady_df_subset))
    gt_key = GT_CATEGORY_MAP.get(category_name)
    if gt_key is None:
        return np.ones(len(steady_df_subset))
    gt_cat = embu_trends_category_index[
        embu_trends_category_index["category"] == gt_key
    ][["month","category_index"]].copy()
    if gt_cat.empty:
        return np.ones(len(steady_df_subset))
    gt_cat = gt_cat.set_index("month")["category_index"]
    weights = []
    for _, row in steady_df_subset.iterrows():
        month = pd.to_datetime(row["months"]).to_period("M").to_timestamp()
        w = gt_cat.get(month, None)
        if w is None: w = gt_cat.iloc[-1] if len(gt_cat) > 0 else 1.0
        weights.append(max(float(w), 0.15))
    return np.array(weights)


def compute_credible_intervals_gt(prediction, steady_df, penetration_factor, confidence_level=0.90):
    credible_intervals = []
    for _, row in prediction.iterrows():
        category = row["correct_therapeutic_class"]
        point    = row["opening_stock_qty"]
        obs_df   = steady_df[
            steady_df["correct_therapeutic_class"] == category
        ][["months","total_qty_dispensed"]].dropna()
        obs = obs_df["total_qty_dispensed"].values

        if len(obs) < 2:
            credible_intervals.append({
                "correct_therapeutic_class": category,
                "ci_lower": max(0, round(point * 0.60)), "ci_upper": round(point * 1.40),
                "ci_method": "fallback ±40%", "n_observations": len(obs)
            }); continue

        weights = compute_gt_weights(category, obs_df)
        w_sum   = weights.sum()
        w_mean  = (weights * obs).sum() / w_sum
        w_var   = (weights * (obs - w_mean) ** 2).sum() / w_sum
        w_std   = np.sqrt(w_var)

        if w_std <= 0 or np.isnan(w_std) or np.isnan(w_mean):
            credible_intervals.append({
                "correct_therapeutic_class": category,
                "ci_lower": max(0, round(point * 0.60)), "ci_upper": round(point * 1.40),
                "ci_method": "fallback ±40% (zero variance)", "n_observations": len(obs)
            }); continue

        mu_m1  = w_mean * penetration_factor
        std_m1 = w_std  * penetration_factor
        alpha  = (1 - confidence_level) / 2
        lo_f   = stats.norm.ppf(alpha, loc=mu_m1, scale=std_m1)
        hi_f   = stats.norm.ppf(1 - alpha, loc=mu_m1, scale=std_m1)

        if np.isnan(lo_f) or np.isnan(hi_f):
            lo, hi, method = max(0, round(point * 0.60)), round(point * 1.40), "fallback ±40% (ppf NaN)"
        else:
            lo, hi = max(0, round(lo_f)), round(hi_f)
            gt_key = GT_CATEGORY_MAP.get(category)
            method = f"GT-weighted normal (n={len(obs)})" if gt_key else f"normal fit (n={len(obs)} months)"

        credible_intervals.append({
            "correct_therapeutic_class": category,
            "ci_lower": lo, "ci_upper": hi, "ci_method": method, "n_observations": len(obs)
        })
    return pd.DataFrame(credible_intervals)


print("✓ compute_credible_intervals_gt() defined")


# Updated external_multiplier ──────────────────────────────────────────────────
def external_multiplier(product_name, embu_target):
    n = str(product_name).lower(); multiplier = 1.0

    # Pharma — hospital proximity
    if embu_target.get("has_hospital_nearby",0) and any(k in n for k in ["injection","injectable","iv ","infusion","vial","ampoule"]): multiplier *= 1.3
    # Malaria
    m = embu_target.get("burden_malaria_ratio",0)
    if m > 0.1 and any(k in n for k in ["artemether","coartem","lumefantrine","quinine","fansidar","sp "]): multiplier *= (1 + m * 2)
    # Maternal
    mb = embu_target.get("burden_maternal_ratio",0); ap = embu_target.get("antenatal_4plus_visits_pct",0)
    if (mb > 0.05 or ap > 0.5) and any(k in n for k in ["folic","ferrous","antenatal","oxytocin","magnesium","prenatal"]): multiplier *= 1.25
    # Paediatric
    pp = embu_target.get("pct_paediatric",0)
    if pp > 0.2 and any(k in n for k in ["syrup","suspension","paediatric","paed","infant","drops","125mg","250mg/5"]): multiplier *= (1 + pp)
    # NCD
    h = embu_target.get("burden_hypertension_ratio",0); d = embu_target.get("burden_diabetes_ratio",0)
    if h > 0.05 and any(k in n for k in ["amlodipine","atenolol","losartan","lisinopril","ramipril","nifedipine"]): multiplier *= (1 + h * 1.5)
    if d > 0.05 and any(k in n for k in ["metformin","glibenclamide","insulin","glucophage"]): multiplier *= (1 + d * 1.5)

    # Beauty
    _is_beauty = any(k in n for k in [
        "serum","moisturiser","moisturizer","sunscreen","spf","cleanser","toner","mask","scrub",
        "lotion","cream","gel","lip balm","lip gloss","body lotion","body butter","face wash",
        "micellar","retinol","vitamin c","niacinamide","hyaluronic","collagen","brightening",
        "whitening","glow","anti-aging","anti-ageing","eye cream","body wash","shower gel",
        "body mist","perfume","cologne","foundation","concealer","mascara","eyeliner","blush","highlighter"
    ])
    if _is_beauty:
        bp = embu_target.get("beauty_demand_proximity",0)
        if bp > 0.3: multiplier *= (1 + bp * 0.5)
        if embu_target.get("beauty_momentum",0) > 0.6: multiplier *= (1 + (embu_target.get("beauty_momentum",0) - 0.5) * 0.4)
        if embu_target.get("beauty_recent4w",0) > 0.7: multiplier *= 1.15
        btp = embu_target.get("beauty_target_pct",0)
        if btp > 0.25: multiplier *= (1 + (btp - 0.25) * 0.8)

    # Body Building
    _is_bb = any(k in n for k in [
        "whey","protein","creatine","pre-workout","preworkout","amino","bcaa","mass gainer",
        "isolate","concentrate","casein","glutamine","beta-alanine","l-carnitine","caffeine",
        "shaker","gym","workout","muscle","bulk","lean mass","fat burner","thermogenic"
    ])
    if _is_bb:
        bbp = embu_target.get("bb_demand_proximity",0)
        if bbp > 0.2: multiplier *= (1 + bbp * 0.8)
        if embu_target.get("bodybuilding_momentum",0) > 0.5: multiplier *= (1 + (embu_target.get("bodybuilding_momentum",0) - 0.5) * 0.5)
        bbtp = embu_target.get("bb_target_pct",0)
        if bbtp > 0.15: multiplier *= (1 + (bbtp - 0.15) * 1.0)

    # Supplements
    _is_supp = any(k in n for k in [
        "vitamin","supplement","zinc","magnesium","calcium","iron","omega","fish oil",
        "multivitamin","probiotic","prebiotic","collagen","biotin","folic acid","vitamin d",
        "vitamin c","vitamin b","immune","antioxidant","melatonin","turmeric","curcumin",
        "glucosamine","chondroitin","coq10","evening primrose","spirulina","moringa","ashwagandha","echinacea"
    ])
    if _is_supp and not _is_bb:
        sp = embu_target.get("supplement_demand_proximity",0)
        if sp > 0.2: multiplier *= (1 + sp * 0.5)
        if embu_target.get("supplements_momentum",0) > 0.5: multiplier *= (1 + (embu_target.get("supplements_momentum",0) - 0.5) * 0.4)
        stp = embu_target.get("supplements_target_pct",0)
        if stp > 0.40: multiplier *= (1 + (stp - 0.40) * 0.6)
        pa = embu_target.get("pct_aged_65_plus",0)
        if pa > 0.05 and any(k in n for k in ["calcium","vitamin d","glucosamine","chondroitin","coq10","omega"]): multiplier *= (1 + pa * 1.5)

    return round(multiplier, 3)


print("✓ external_multiplier() defined with GT + proximity signals")


# ── STEP 3f: Category ↔ Burden mapping ────────────────────────────────────────
diag_monthly = (
    diag_df.groupby(["facility_id","monthly"])["diagnosis_disease_group"]
    .apply(list).reset_index()
)
disp_monthly = (
    disp_df.groupby(["facility_id","months"])["correct_therapeutic_class"]
    .apply(lambda x: list(x.dropna())).reset_index()
    .rename(columns={"months":"monthly"})
)
paired = diag_monthly.merge(disp_monthly, on=["facility_id","monthly"], how="inner")

CATEGORY_CONTEXT = {}
BURDEN_SIGNALS   = {}
all_categories   = disp_df["correct_therapeutic_class"].dropna().unique()

for category in all_categories:
    months_with_cat = paired[paired["correct_therapeutic_class"].apply(lambda cats: category in cats)]
    if months_with_cat.empty:
        CATEGORY_CONTEXT[category] = "Insufficient co-occurrence data"; continue
    all_diags  = [d for row in months_with_cat["diagnosis_disease_group"] for d in row if pd.notna(d)]
    top_labels = [d[0] for d in Counter(all_diags).most_common(3)]
    matched    = [bn.replace("_"," ").title() for bn, kws in BURDEN_MAP.items()
                  if any(any(kw.lower() in diag.lower() for kw in kws) for diag in top_labels)]
    CATEGORY_CONTEXT[category] = " / ".join(matched) if matched else (top_labels[0] if top_labels else "Review therapeutic class data")

for burden_name, keywords in BURDEN_MAP.items():
    months_with_burden = paired[paired["diagnosis_disease_group"].apply(
        lambda diags: any(any(kw.lower() in d.lower() for kw in keywords) for d in diags if pd.notna(d))
    )]
    if months_with_burden.empty:
        BURDEN_SIGNALS[burden_name] = []; continue
    all_cats = [c for row in months_with_burden["correct_therapeutic_class"] for c in row if pd.notna(c)]
    BURDEN_SIGNALS[burden_name] = [c for c, _ in Counter(all_cats).most_common(4)]

print(f"CATEGORY_CONTEXT: {len(CATEGORY_CONTEXT)} categories | BURDEN_SIGNALS: {len(BURDEN_SIGNALS)} dimensions")


# ── 4. BUILD EMBU TARGET PROFILE ──────────────────────────────────────────────
external_feature_names = list(embu_external.keys())
for feat in external_feature_names:
    facility_profiles[feat] = 0

# Scale DHS % values to 0–1
for feat in external_feature_names:
    val = embu_external[feat]
    if val is not None and isinstance(val, (int, float)) and val > 1:
        embu_external[feat] = val / 100

ratio_cols = list(dict.fromkeys(ratio_cols + external_feature_names))

weights     = facility_profiles["avg_monthly_consultations"].fillna(1)
embu_target = facility_profiles[ratio_cols].apply(lambda col: (col * weights).sum() / weights.sum())

for feat, val in embu_external.items():
    embu_target[feat] = val if val is not None else 0

embu_target["n_hospitals_25km"]    = (nearby_all["amenity"] == "hospital").sum()
embu_target["n_clinics_25km"]      = nearby_all["amenity"].isin(["clinic","doctors"]).sum()
embu_target["n_pharmacies_25km"]   = (nearby_all["amenity"] == "pharmacy").sum()
embu_target["n_facilities_25km"]   = len(nearby_all)
embu_target["has_hospital_nearby"] = 1

for col in cat_ratio_cols:
    embu_target[col] = facility_profiles[col].mean()

print("Embu catchment profile — top disease burden dimensions:")
burden_ratios = {k.replace("burden_","").replace("_ratio",""): round(v,3) for k,v in embu_target.items() if "burden_" in k and v > 0}
for name, score in sorted(burden_ratios.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:<20} {score:.3f}  {'█' * int(score*40)}")


# ── 5. STEADY-STATE DISPENSING ────────────────────────────────────────────────
disp_work = disp_df.copy()
disp_work["month_rank"] = disp_work.groupby("facility_id")["months"].rank(method="dense").astype(int)

month1 = (
    disp_work[disp_work["month_rank"] == 1]
    .groupby(["facility_id","correct_therapeutic_class"])
    .agg(month1_qty=("total_qty_dispensed","sum"))
    .reset_index()
)

steady_df = disp_work[disp_work["month_rank"] > RAMP_UP_MONTHS]
if steady_df.empty:
    print(f"WARNING: No data beyond month {RAMP_UP_MONTHS}. Using all months.")
    steady_df = disp_work.copy()

steady_agg = (
    steady_df
    .groupby(["facility_id","correct_therapeutic_class"])
    .agg(avg_monthly_qty=("total_qty_dispensed","mean"),
         avg_unique_patients=("unique_patients","mean"),
         months_of_data=("months","nunique"))
    .reset_index()
)
print(f"Steady-state: {len(steady_agg)} rows | {steady_agg['correct_therapeutic_class'].nunique()} categories")


# ── 6. PENETRATION FACTOR ─────────────────────────────────────────────────────
merged = steady_agg.merge(month1, on=["facility_id","correct_therapeutic_class"], how="inner")
merged["penetration_ratio"] = merged["month1_qty"] / merged["avg_monthly_qty"].replace(0, np.nan)
penetration_factor = merged["penetration_ratio"].dropna().median()
pf_is_default = False

if pd.isna(penetration_factor) or penetration_factor <= 0 or penetration_factor > 2:
    print("WARNING: Cannot compute reliable penetration factor. Falling back to 0.35")
    penetration_factor = 0.35; pf_is_default = True
else:
    print(f"Penetration factor: {penetration_factor:.2f} ({penetration_factor*100:.0f}% of mature monthly volume in Month 1)")


# ── 7. KNN FACILITY SIMILARITY ────────────────────────────────────────────────
profiles_indexed = facility_profiles.set_index("facility_id")
X = profiles_indexed[ratio_cols].fillna(0)
print(f"Feature matrix: {X.shape[0]} facilities × {X.shape[1]} features")

scaler   = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

embu_vector = embu_target.reindex(ratio_cols).fillna(0).values.reshape(1, -1)
embu_scaled = scaler.transform(embu_vector)[0]

n_facilities = len(X_scaled)
k = min(KNN_K, n_facilities)

if n_facilities < 2:
    weight_map = {X_scaled.index[0]: 1.0}; used_fallback = False
else:
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(X_scaled)
    distances, indices = knn.kneighbors([embu_scaled])
    distances, indices = distances[0], indices[0]
    top_facility_ids = [X_scaled.index[i] for i in indices]

    print(f"\nTop {k} facilities most similar to Embu profile:")
    for fid, dist in zip(top_facility_ids, distances):
        interp = "Very similar" if dist < 0.1 else "Similar" if dist < 0.3 else "Moderately similar"
        print(f"  facility_id={fid}  distance={dist:.4f}  {interp}")

    spread = distances.max() - distances.min()
    if spread < KNN_FALLBACK_THRESHOLD:
        print("Spread too narrow → using equal weights (fallback).")
        weight_map = {fid: 1.0 / n_facilities for fid in X_scaled.index}; used_fallback = True
    else:
        weights_knn = 1 / (distances + 1e-6); weights_knn /= weights_knn.sum()
        weight_map = dict(zip(top_facility_ids, weights_knn)); used_fallback = False
        for fid, w in weight_map.items():
            print(f"  facility_id={fid}  weight={w:.3f}  ({w*100:.1f}%)")


# ── 8. PREDICT OPENING STOCK ──────────────────────────────────────────────────
df_pred = steady_agg[steady_agg["facility_id"].isin(weight_map.keys())].copy()
df_pred["weight"]       = df_pred["facility_id"].map(weight_map).fillna(0)
df_pred["weighted_qty"] = df_pred["avg_monthly_qty"] * df_pred["weight"]

prediction = (
    df_pred.groupby("correct_therapeutic_class")
    .agg(predicted_steady_state=("weighted_qty","sum"),
         avg_unique_patients=("avg_unique_patients","mean"),
         n_facilities_contributing=("facility_id","nunique"))
    .reset_index()
)

prediction["opening_stock_qty"] = (prediction["predicted_steady_state"] * penetration_factor).round().astype(int)
prediction["dead_stock_risk"]   = prediction["opening_stock_qty"] > prediction["predicted_steady_state"] * DEAD_STOCK_MULTIPLIER

months_avg = steady_agg.groupby("correct_therapeutic_class")["months_of_data"].mean().reset_index().rename(columns={"months_of_data":"months_avg"})
prediction = prediction.merge(months_avg, on="correct_therapeutic_class", how="left")

prediction["confidence"] = prediction.apply(
    lambda row: "Low"    if pf_is_default or row["n_facilities_contributing"] < 2
                else "High"   if not used_fallback and row.get("months_avg",0) >= 6
                else "Medium", axis=1)

prediction["diagnosis_driver"] = prediction["correct_therapeutic_class"].map(CATEGORY_CONTEXT).fillna("Review therapeutic class data")

top_burden = sorted(
    [(k.replace("burden_","").replace("_ratio",""), v) for k, v in embu_target.items() if "burden_" in k and v > 0],
    key=lambda x: x[1], reverse=True
)[:3]


def build_reasoning(row):
    reasons = []; category = row["correct_therapeutic_class"]
    drivers = category_proximity_driver.get(category, [])
    if drivers: reasons.append(f"{len(drivers)} nearby {'facility' if len(drivers)==1 else 'facilities'} — {drivers[0]}")
    for bn, bs in top_burden:
        if category in BURDEN_SIGNALS.get(bn,[]) and bs > 0.1:
            reasons.append(f"High {bn} catchment burden ({bs:.0%})"); break
    if embu_target.get("pct_chronic_age",0) > 0.25 and category == "Oral Solid Forms":
        reasons.append(f"High adult 45+ population ({embu_target['pct_chronic_age']:.0%})")
    if embu_target.get("pct_paediatric",0) > 0.20 and category in ["Oral Liquid Forms","Vaccines & Biologicals"]:
        reasons.append(f"High paediatric population ({embu_target['pct_paediatric']:.0%})")
    if not used_fallback and row["n_facilities_contributing"] >= 2:
        reasons.append(f"Comparable to Branch {max(weight_map, key=weight_map.get)} Month 1 pattern")
    if row["dead_stock_risk"]: return "Low cross-branch demand — order conservatively"
    if row["confidence"] == "Low": reasons.append("Insufficient history — conservative estimate")
    return " — ".join(reasons[:2]) if reasons else "Based on Embu catchment profile"


prediction["model_reasoning"] = prediction.apply(build_reasoning, axis=1)
print(f"Prediction: {len(prediction)} categories")
print(prediction[["correct_therapeutic_class","opening_stock_qty","confidence"]].to_string(index=False))


# ── 9. BAYESIAN CI (GT-WEIGHTED) ──────────────────────────────────────────────
ci_df = compute_credible_intervals_gt(prediction, steady_df, penetration_factor, CONFIDENCE_LEVEL)

prediction = prediction.merge(ci_df, on="correct_therapeutic_class", how="left")
prediction["stock_range"] = (
    prediction["ci_lower"].astype(int).astype(str) + " – " +
    prediction["ci_upper"].astype(int).astype(str) + f" ({int(CONFIDENCE_LEVEL*100)}% CI)"
)

print("Credible intervals computed:")
print(prediction[["correct_therapeutic_class","opening_stock_qty","stock_range","ci_method"]].to_string(index=False))


# ── 10. FINAL OUTPUT ──────────────────────────────────────────────────────────
output = prediction[[
    "correct_therapeutic_class","opening_stock_qty","stock_range",
    "predicted_steady_state","confidence","dead_stock_risk","model_reasoning",
]].sort_values("opening_stock_qty", ascending=False).copy()

output.columns = [
    "Category","Opening Stock Qty","Stock Range (90% CI)",
    "Predicted Monthly (Steady State)","Confidence","Dead Stock Risk","Model Reasoning"
]

print("=" * 70)
print("BRANCH #106 — RECOMMENDED OPENING STOCK")
print(f"Method:             {'KNN-weighted' if not used_fallback else 'equal-weight average'}")
print(f"Penetration factor: {penetration_factor:.2f}" + (" (default)" if pf_is_default else " (data-derived)"))
print(f"Facilities used:    {len(weight_map)}")
print("=" * 70)
print(output.to_string(index=False))


# ── STEP 9b: PRODUCT-LEVEL ALLOCATION ────────────────────────────────────────
prod_steady = (
    disp_work[
        (disp_work["month_rank"] > RAMP_UP_MONTHS) &
        (disp_work["facility_id"].isin(weight_map))
    ].copy()
)
prod_steady["weight"]       = prod_steady["facility_id"].map(weight_map)
prod_steady["weighted_qty"] = prod_steady["total_qty_dispensed"] * prod_steady["weight"]

# ── FIX: Pull product names from BOTH fact_dispensing AND product_intel ───────
# fact_dispensing covers real pharma products.
# product_intel covers simulated new-category products (Beauty, Supplements, BB).
# Without product_intel here, new-category products have no name → get dropped.
prod_names_real = fact_dispensing[["product_id","product_name"]].dropna().drop_duplicates("product_id")

if (not product_intel.empty
        and "product_id" in product_intel.columns
        and "product_name" in product_intel.columns):
    prod_names_new = product_intel[["product_id","product_name"]].dropna().drop_duplicates("product_id")
    prod_names = pd.concat([prod_names_real, prod_names_new], ignore_index=True).drop_duplicates("product_id")
    print(f"prod_names: {len(prod_names)} products "
          f"({len(prod_names_real)} real + {len(prod_names_new)} simulated new-category)")
else:
    prod_names = prod_names_real
    print(f"prod_names: {len(prod_names)} products (product_intel not available — new categories may be missing)")
# ─────────────────────────────────────────────────────────────────────────────

prod_agg = (
    prod_steady.merge(prod_names, on="product_id", how="left")
    .groupby(["correct_therapeutic_class","product_id","product_name"])
    .agg(weighted_qty=("weighted_qty","sum"))
    .reset_index()
)
prod_agg = prod_agg[prod_agg["product_name"].notna()]

cat_total_qty          = prod_agg.groupby("correct_therapeutic_class")["weighted_qty"].transform("sum")
prod_agg["base_share"] = prod_agg["weighted_qty"] / cat_total_qty.replace(0, 1)

prod_agg["ext_multiplier"] = prod_agg["product_name"].apply(lambda n: external_multiplier(n, embu_target))

prod_agg["adjusted_weight"] = prod_agg["base_share"] * prod_agg["ext_multiplier"]
adj_cat_total               = prod_agg.groupby("correct_therapeutic_class")["adjusted_weight"].transform("sum")
prod_agg["adjusted_share"]  = prod_agg["adjusted_weight"] / adj_cat_total.replace(0, 1)

cat_opening = output.set_index("Category")["Opening Stock Qty"]
prod_agg["category_opening_qty"] = prod_agg["correct_therapeutic_class"].map(cat_opening)
prod_agg["product_opening_qty"]  = (prod_agg["adjusted_share"] * prod_agg["category_opening_qty"]).round().astype(int)
prod_agg = prod_agg[prod_agg["product_opening_qty"] > 0]

prod_agg["confidence"]     = prod_agg["correct_therapeutic_class"].map(output.set_index("Category")["Confidence"])
prod_agg["dead_stock_risk"] = prod_agg["correct_therapeutic_class"].map(output.set_index("Category")["Dead Stock Risk"])

output_products = (
    prod_agg[[
        "product_name","correct_therapeutic_class","product_opening_qty",
        "base_share","ext_multiplier","confidence","dead_stock_risk"
    ]]
    .sort_values(["correct_therapeutic_class","product_opening_qty"], ascending=[True, False])
    .reset_index(drop=True)
)
output_products.columns = [
    "Product","Category","Opening Stock Qty",
    "Historical Share","External Adjustment","Confidence","Dead Stock Risk"
]
output_products["Historical Share"]    = (output_products["Historical Share"] * 100).round(1).astype(str) + "%"
output_products["External Adjustment"] = output_products["External Adjustment"].apply(lambda x: f"↑ {x:.2f}x" if x > 1 else f"→ {x:.2f}x")
output_products["Dead Stock Risk"]     = output_products["Dead Stock Risk"].map({True:"⚠️ Yes", False:"✓ No"}).fillna(output_products["Dead Stock Risk"])

print(f"\nProduct-level output: {len(output_products)} products across {output_products['Category'].nunique()} categories")

# Verify new categories made it through
for cat in ["Beauty Products", "Vitamins & Supplements", "Body Building"]:
    n = (output_products["Category"] == cat).sum()
    print(f"  {'✓' if n > 0 else '✗ MISSING'} {cat}: {n} products")

print("\nTop 20 by opening stock qty:")
print(output_products.nlargest(20, "Opening Stock Qty")[
    ["Product","Category","Opening Stock Qty","Historical Share","External Adjustment","Confidence"]
].to_string(index=False))


# ── SAVE OUTPUTS ──────────────────────────────────────────────────────────────
pickle.dump({
    "disp":          fact_dispensing,
    "inv":           fact_inventory_snapshot,
    "pat":           dim_patient_profile,
    "diag_df":       monthly_diagnoses_aggregate,
    "disp_df":       monthly_dispensing_aggregate,
    "pred":          output,
    "pred_products": output_products,
    "product_intel": product_intel,
}, open("data_export.pkl", "wb"))

output_products.to_csv("branch_106_opening_stock_products.csv", index=False)

print("\n✓ Saved → data_export.pkl")
print("✓ Saved → branch_106_opening_stock_products.csv")
print(f"  Pickle keys: disp, inv, pat, diag_df, disp_df, pred, pred_products, product_intel")