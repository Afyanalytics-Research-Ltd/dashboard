"""
geocode_locations.py: fill HOSPITALS.LOOKUP_TABLES.NANDI_GEO_COORDINATES for every county the
hospital sees patients from (not just Nandi). Runs on your machine through the same
SnowflakeClient (.env + private key) the dashboard uses.

    python tools/geocode_locations.py --report            # what is covered, what is missing
    python tools/geocode_locations.py --seed              # add missing places from the outpatient data as PENDING
    python tools/geocode_locations.py --run               # geocode PENDING rows (Nominatim)
    python tools/geocode_locations.py --seed --run        # both
    python tools/geocode_locations.py --retry-not-found --run
    python tools/geocode_locations.py --run --limit 50    # small test first

Needs: pip install aiohttp  (snowflake connector + python-dotenv already come with the dashboard env)
Nominatim's policy is max 1 request per second, so the default is one request per second.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in HERE.parents:
    if (p / "snowflake_service").exists():
        sys.path.insert(0, str(p))
        break
from snowflake_service.snowflake_client import SnowflakeClient  # noqa: E402

GEO_TABLE = "HOSPITALS.LOOKUP_TABLES.NANDI_GEO_COORDINATES"
SRC_TABLE = "HOSPITALS.NANDI_CLEAN.NANDI_OUTPATIENT_WIDE"
INTERNATIONAL = ("Rwanda", "United Kingdom", "Uganda")
INTL_SQL = ", ".join(f"'{c}'" for c in INTERNATIONAL)
NOMINATIM = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "afya_analytics_kapsabet_geocoder_v3 (mercykubz@gmail.com)"
DELAY = 1.1  # seconds between requests
LEVEL_ORDER = "CASE geo_level WHEN 'COUNTY' THEN 1 WHEN 'SUB_COUNTY' THEN 2 WHEN 'WARD' THEN 3 ELSE 4 END"


def run(client: SnowflakeClient, sql: str, params=None):
    """Statements that write (the client's query() is read-only: it fetches a DataFrame)."""
    cur = client.conn.cursor()
    try:
        cur.execute(sql, params) if params else cur.execute(sql)
        return cur.rowcount
    finally:
        cur.close()


# ------------------------------------------------------------------ report
def report(client):
    df = client.query(f"""
        SELECT geo_level, COALESCE(county,'') AS county_group, geocode_status, COUNT(*) AS n
        FROM {GEO_TABLE} GROUP BY 1, 2, 3 ORDER BY {LEVEL_ORDER}, 2""")
    df.columns = [c.lower() for c in df.columns]
    print("\nRows in the geo table by level and status")
    print(df.groupby(["geo_level", "geocode_status"])["n"].sum().unstack(fill_value=0).to_string())
    missing = client.query(f"""
        WITH src AS (
          SELECT 'COUNTY' AS geo_level, county AS geo_name, NULL AS parent_name, county, COUNT(*) AS visits
          FROM {SRC_TABLE} WHERE county IS NOT NULL AND county NOT IN ({INTL_SQL}) GROUP BY county
          UNION ALL
          SELECT 'SUB_COUNTY', sub_county, county, county, COUNT(*) FROM {SRC_TABLE}
          WHERE sub_county IS NOT NULL AND county IS NOT NULL AND county NOT IN ({INTL_SQL}) GROUP BY sub_county, county
          UNION ALL
          SELECT 'WARD', ward, sub_county, county, COUNT(*) FROM {SRC_TABLE}
          WHERE ward IS NOT NULL AND county IS NOT NULL AND county NOT IN ({INTL_SQL}) GROUP BY ward, sub_county, county
          UNION ALL
          SELECT 'LOCALITY', locality, ward, county, COUNT(*) FROM {SRC_TABLE}
          WHERE locality IS NOT NULL AND county IS NOT NULL AND county NOT IN ({INTL_SQL}) GROUP BY locality, ward, county)
        SELECT s.geo_level, COUNT(*) AS places_not_in_table, SUM(s.visits) AS visits_affected
        FROM src s LEFT JOIN {GEO_TABLE} g
          ON g.geo_level = s.geo_level AND UPPER(TRIM(g.geo_name)) = UPPER(TRIM(s.geo_name))
         AND UPPER(TRIM(COALESCE(g.county,''))) = UPPER(TRIM(COALESCE(s.county,'')))
        WHERE g.geo_name IS NULL GROUP BY s.geo_level""")
    print("\nPlaces seen in patient data but missing from the geo table (run --seed to add them)")
    print(missing.to_string(index=False) if len(missing) else "none")


# ------------------------------------------------------------------ seed
def seed(client):
    """Insert every county / sub-county / ward / locality in the patient data that the table lacks.
    Parent: sub-county -> county, ward -> sub-county, locality -> ward (the nesting the geocoder queries use)."""
    parts = [  # level, name column, parent column
        ("COUNTY", "county", None),
        ("SUB_COUNTY", "sub_county", "county"),
        ("WARD", "ward", "sub_county"),
        ("LOCALITY", "locality", "ward"),
    ]
    total = 0
    for level, col, parent in parts:
        sql = f"""
        INSERT INTO {GEO_TABLE} (GEO_LEVEL, GEO_NAME, PARENT_NAME, COUNTY, GEOCODE_STATUS)
        SELECT DISTINCT '{level}', s.{col}, {'s.' + parent if parent else 'NULL'}, s.county, 'PENDING'
        FROM {SRC_TABLE} s
        WHERE s.county IS NOT NULL AND s.county NOT IN ({INTL_SQL})
          AND s.geo_confidence IN ('High', 'Medium') AND s.{col} IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM {GEO_TABLE} g
            WHERE g.geo_level = '{level}' AND UPPER(TRIM(g.geo_name)) = UPPER(TRIM(s.{col}))
              AND UPPER(TRIM(COALESCE(g.county, ''))) = UPPER(TRIM(s.county)))"""
        n = run(client, sql)
        print(f"  seeded {n or 0:>5} {level} rows")
        total += n or 0
    print(f"Seeded {total} new PENDING rows")


# ------------------------------------------------------------------ geocode
def queries_for(level, name, parent, county):
    parent, county = parent or "", county or ""
    if level == "COUNTY":
        return [f"{name} County, Kenya"]
    if level == "SUB_COUNTY":
        return [f"{name} sub-county, {county} County, Kenya", f"{name}, {county} County, Kenya"]
    if level == "WARD":
        return [f"{name} ward, {parent}, {county} County, Kenya", f"{name}, {county} County, Kenya"]
    return [f"{name}, {parent}, {county} County, Kenya", f"{name}, {county} County, Kenya"]


async def hit(session, q, polygon):
    params = {"q": q, "format": "json", "limit": "1", "accept-language": "en", "countrycodes": "ke"}
    if polygon:
        params.update({"polygon_geojson": "1", "polygon_threshold": "0.001"})  # simplified outline keeps the payload small
    for attempt in range(3):
        try:
            async with session.get(NOMINATIM, params=params, headers={"User-Agent": USER_AGENT}) as r:
                if r.status == 200:
                    d = await r.json()
                    if d:
                        h = d[0]
                        gj = h.get("geojson")
                        if gj and gj.get("type") not in ("Polygon", "MultiPolygon"):
                            gj = None  # a point or line is not a boundary
                        return float(h["lat"]), float(h["lon"]), (json.dumps(gj, separators=(",", ":")) if gj else None)
                    return None
                if r.status in (429, 503):
                    await asyncio.sleep(5 * (attempt + 1))
        except Exception as e:  # network blip: retry
            print(f"   error: {e}")
            await asyncio.sleep(2)
    return None


def save(client, rows):
    for r in rows:
        run(client, f"""UPDATE {GEO_TABLE}
            SET LATITUDE=%s, LONGITUDE=%s, BOUNDARY_GEOJSON=%s, GEOCODE_STATUS=%s
            WHERE GEO_LEVEL=%s AND GEO_NAME=%s AND COALESCE(PARENT_NAME,'')=COALESCE(%s,'') AND COALESCE(COUNTY,'')=COALESCE(%s,'')""",
            (r["lat"], r["lon"], r["gj"], r["status"], r["level"], r["name"], r["parent"], r["county"]))


async def geocode(client, limit):
    df = client.query(f"""SELECT geo_level, geo_name, parent_name, county FROM {GEO_TABLE}
                          WHERE geocode_status = 'PENDING' ORDER BY {LEVEL_ORDER}, county, geo_name""")
    df.columns = [c.lower() for c in df.columns]
    if limit:
        df = df.head(limit)
    total = len(df)
    print(f"{total} pending rows (about {total * DELAY / 60:.0f} min at 1 request/second, more when the first query misses)")
    if not total:
        return
    import aiohttp

    done, found, buf, t0 = 0, 0, [], time.time()
    async with aiohttp.ClientSession() as session:
        for row in df.itertuples(index=False):
            res = None
            for q in queries_for(row.geo_level, row.geo_name, row.parent_name, row.county):
                res = await hit(session, q, row.geo_level != "LOCALITY")
                await asyncio.sleep(DELAY)
                if res:
                    break
            lat, lon, gj = res if res else (None, None, None)
            buf.append(dict(lat=lat, lon=lon, gj=gj, status="FOUND" if res else "NOT_FOUND",
                            level=row.geo_level, name=row.geo_name, parent=row.parent_name, county=row.county))
            found += bool(res)
            done += 1
            if len(buf) >= 25 or done == total:
                save(client, buf)
                buf = []
                print(f"  {done}/{total} done, {found} found, {time.time() - t0:.0f}s")
    print(f"Finished: {found}/{total} geocoded. Anything NOT_FOUND stays in the table for a manual fix.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--retry-not-found", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if not (a.report or a.seed or a.run or a.retry_not_found):
        ap.print_help()
        return
    client = SnowflakeClient()
    if a.report:
        report(client)
    if a.seed:
        seed(client)
    if a.retry_not_found:
        print(f"Reset {run(client, f'''UPDATE {GEO_TABLE} SET GEOCODE_STATUS='PENDING' WHERE GEOCODE_STATUS='NOT_FOUND' ''')} rows to PENDING")
    if a.run:
        asyncio.run(geocode(client, a.limit))
        report(client)


if __name__ == "__main__":
    main()
