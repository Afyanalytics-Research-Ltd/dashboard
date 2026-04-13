"""
Pharmaplus Proximity Scraper — SerpAPI Google Maps
----------------------------------------------------
For each pharmacy location, queries nearby places across
beauty, gym, supplement, pharmacy, and supermarket categories.
Flags chain vs. independent businesses.
Output: one combined CSV — pharmaplus_proximity_data.csv

Usage:
    pip install serpapi pandas
    python pharmaplus_proximity_scraper.py
"""

import os

import serpapi
import pandas as pd
import math
import time
from dotenv import load_dotenv
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

LOCATIONS = [
    {
        "facility_id": 4,
        "name": "Pharmaplus (Tenri Maternity & Theatre Site)",
        "lat": -0.542489,
        "lon":  37.465686,
        "type": "Pharmacy",
    },
    {
        "facility_id": 5,
        "name": "Pharmaplus (Tenri General Hospital Site)",
        "lat": -0.542050,
        "lon":  37.465775,
        "type": "Pharmacy",
    },
]

SEARCH_QUERIES = [
    {"category": "gym_fitness",           "query": "gym fitness center bodybuilding"},
    {"category": "beauty_salon_spa",      "query": "beauty salon spa"},
    {"category": "beauty_shop_cosmetics", "query": "beauty shop cosmetics"},
    {"category": "supplements_vitamins",  "query": "vitamins supplements health store"},
    {"category": "bodybuilding_shop",     "query": "bodybuilding supplement shop"},
    {"category": "pharmacy_competitor",   "query": "pharmacy"},
    {"category": "supermarket",           "query": "supermarket"},
    {"category": "university_college",  "query": "university"},
    {"category": "university_college",  "query": "college"},
    {"category": "university_college",  "query": "polytechnic"},
    {"category": "university_college",  "query": "technical institute"},
    {"category": "university_college",  "query": "KMTC"},
]

SEARCH_RADIUS_METERS = 10000

# ── CHAIN LISTS ───────────────────────────────────────────────────────────────

SUPERMARKET_CHAINS = [
    "Naivas", "Carrefour", "QuickMart", "Quickmart", "Zucchini",
    "Eastmatt", "Cleanshelf", "Uchumi", "Tuskys", "Spar",
    "Game", "Shoprite", "Chandarana", "Massmart", "Choppies",
    "Hotpoint", "Mulleys", "Mullies",
]

PHARMACY_CHAINS = [
    "Goodlife", "Good Life", "Medisel", "Portal", "Haltons",
    "Faraja", "Pharmaken", "HealthPlus", "Health Plus", "Medicare",
    "Dawa", "Peoples", "People's", "Ladnan", "Citylife", "City Life",
    "Pharmaplus", "Lifecare", "Life Care", "Meru Pharmacy",
]

BEAUTY_CHAINS = [
    "Revlon", "L'Oreal", "Loreal", "Nairobi Beauty", "Style Avenue",
    "Supa Beauty", "House of Beauty",
]

GYM_CHAINS = [
    "Gymkhana", "Imax", "Planet Fitness", "Virgin Active",
    "Ken Gym", "Flexx", "Athlete's",
]

CHAIN_LOOKUP = (
    [(name, "supermarket_chain") for name in SUPERMARKET_CHAINS]
    + [(name, "pharmacy_chain")  for name in PHARMACY_CHAINS]
    + [(name, "beauty_chain")    for name in BEAUTY_CHAINS]
    + [(name, "gym_chain")       for name in GYM_CHAINS]
)

# ── CHAIN DETECTION ───────────────────────────────────────────────────────────

def detect_chain(place_name: str):
    name_lower = place_name.lower()
    for chain, chain_cat in CHAIN_LOOKUP:
        if chain.lower() in name_lower:
            return True, chain, chain_cat
    return False, "", ""

# ── HAVERSINE DISTANCE ────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2  = math.radians(lat1), math.radians(lat2)
    dphi        = math.radians(lat2 - lat1)
    dlambda     = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ── SCRAPER ───────────────────────────────────────────────────────────────────

def query_serpapi(facility, query, category):
    lat, lon = facility["lat"], facility["lon"]

    results = serpapi.GoogleSearch({
        "engine":  "google_maps",
        "q":       query,
        "ll":      f"@{lat},{lon},14z",
        "type":    "search",
        "radius":  SEARCH_RADIUS_METERS,
        "hl":      "en",
        "gl":      "ke",
        "api_key": SERPAPI_KEY,
    }).get_dict()

    local_results = results.get("local_results", [])

    rows = []
    for place in local_results:
        gps        = place.get("gps_coordinates", {})
        place_lat  = gps.get("latitude", None)
        place_lon  = gps.get("longitude", None)
        place_name = place.get("title", "")

        distance_km = (
            round(haversine_km(lat, lon, place_lat, place_lon), 3)
            if place_lat and place_lon else ""
        )

        is_chain, chain_name, chain_category = detect_chain(place_name)

        rows.append({
            "facility_id":    facility["facility_id"],
            "facility_name":  facility["name"],
            "facility_type":  facility["type"],
            "facility_lat":   lat,
            "facility_lon":   lon,
            "category":       category,
            "search_query":   query,
            "place_name":     place_name,
            "place_address":  place.get("address", ""),
            "place_lat":      place_lat,
            "place_lon":      place_lon,
            "distance_km":    distance_km,
            "is_chain":       is_chain,
            "chain_name":     chain_name,
            "chain_category": chain_category,
            "rating":         place.get("rating", ""),
            "reviews_count":  place.get("reviews", ""),
            "place_type":     ", ".join(place.get("type", [])) if isinstance(place.get("type"), list) else place.get("type", ""),
            "phone":          place.get("phone", ""),
            "website":        place.get("website", ""),
            "open_now":       place.get("open_state", ""),
        })

    return rows


def run_scraper():
    all_results = []
    total = len(LOCATIONS) * len(SEARCH_QUERIES)
    count = 0

    for facility in LOCATIONS:
        for sq in SEARCH_QUERIES:
            count += 1
            print(f"[{count}/{total}] {facility['name']} — {sq['category']} ...")
            try:
                rows = query_serpapi(facility, sq["query"], sq["category"])
                all_results.extend(rows)
                print(f"         → {len(rows)} places found")
            except Exception as e:
                print(f"         ✗ Error: {e}")
            time.sleep(1)

    df = pd.DataFrame(all_results)

    if df.empty:
        print("\nNo results returned. Check your API key and coordinates.")
        return

    df = df.sort_values(["facility_id", "category", "distance_km"]).reset_index(drop=True)

    output_path = "tendri\\data\\pharmaplus_proximity_data.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Done. {len(df)} total places saved to: {output_path}")

    print("\nPlaces found by facility × category:")
    summary = df.groupby(["facility_name", "category"]).size().reset_index(name="count")
    print(summary.to_string(index=False))

    chains = df[df["is_chain"] == True][["facility_name", "category", "place_name", "chain_name", "chain_category", "distance_km"]]
    if not chains.empty:
        print(f"\nChain businesses detected ({len(chains)} total):")
        print(chains.to_string(index=False))
    else:
        print("\nNo known chain businesses detected in results.")

    print("\nNearest place per facility × category:")
    nearest = df[df["distance_km"] != ""].copy()
    nearest["distance_km"] = nearest["distance_km"].astype(float)
    print(
        nearest.loc[
            nearest.groupby(["facility_id", "category"])["distance_km"].idxmin(),
            ["facility_name", "category", "place_name", "is_chain", "distance_km"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    run_scraper()