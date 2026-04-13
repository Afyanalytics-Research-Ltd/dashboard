"""
Embu 2019 Census & CIDP Population Extractor
==============================================
Extracts Embu county population data from two PDFs using fitz (PyMuPDF).

Sources:
  VOL1_PDF — 2019-Kenya-population-and-Housing-Census-Volume-1-Population-By-County-And-Sub-County.pdf
  CIDP_PDF — EMBU_CIDP_2023-2027.pdf

Outputs (saved to OUT_DIR):
  embu_table25_sex_by_subcounty.csv         Table 2.5: male / female / intersex / total per sub-county
  embu_table26_households_by_subcounty.csv  Table 2.6: population / households / avg_hh_size per sub-county
  embu_table27_density_by_subcounty.csv     Table 2.7: population / land_area_km2 / density per sub-county
  embu_cidp_age_cohort_projections.csv      CIDP Table 5: 17 age cohorts x M/F/T for 2019/2022/2025/2027
  embu_cidp_urban_area_projections.csv      CIDP Table 6: 10 urban centres x M/F/T for 2019/2022/2025/2027
  embu_cidp_ward_density_projections.csv    CIDP Table 7c: 21 wards x population/density for 2019/2022/2025/2027
  embu_cidp_broad_age_projections.csv       CIDP Table 8: 8 broad age groups x M/F/T for 2019/2022/2025/2027

Requirements:
  pip install pymupdf pandas
"""

import fitz        # pip install pymupdf
import pandas as pd
import re
import os

# ── Update these paths ────────────────────────────────────────────────────────
VOL1_PDF = r"C:\Users\Mercy\Documents\Tendri\Snowflake Pulls\Xana\snowflake\tendri\data\2019-Kenya-population-and-Housing-Census-Volume-1-Population-By-County-And-Sub-County.pdf"
CIDP_PDF = r"C:\Users\Mercy\Documents\Tendri\Snowflake Pulls\Xana\snowflake\tendri\data\EMBU_CIDP_2023-2027.pdf"
OUT_DIR  = r"tendri\data"
os.makedirs(OUT_DIR, exist_ok=True)

YEARS = [2019, 2022, 2025, 2027]

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_num(s):
    try: return int(str(s).replace(",", "").strip())
    except: return None

def parse_float(s):
    try: return float(str(s).replace(",", "").strip())
    except: return None

def clean_name(s):
    """Strip trailing dots/ellipses/whitespace from place names."""
    return re.sub(r"[.…\s]+$", "", s).strip()

def get_lines(doc, page_idx):
    """Extract non-empty stripped lines from a PDF page."""
    return [l.strip() for l in doc[page_idx].get_text().split("\n") if l.strip()]

def find_embu_block(lines, end_marker="Kitui..…"):
    """Find the start and end line indices of the Embu county block."""
    start = next(i for i, l in enumerate(lines) if "Embu..…" in l)
    end   = next(i for i, l in enumerate(lines) if i > start and end_marker in l)
    return start, end


# ══════════════════════════════════════════════════════════════════════════════
# VOLUME I — TABLE 2.5
# Population by Sex and Sub-County
# PDF page 24 (0-indexed 23)
# Columns: level | name | male | female | intersex | total
# ══════════════════════════════════════════════════════════════════════════════
print("Extracting Table 2.5 — Population by Sex...")

doc_vol1 = fitz.open(VOL1_PDF)
lines = get_lines(doc_vol1, 23)
start, end = find_embu_block(lines)

rows = []
i = start
while i < end:
    line = lines[i]
    if line == ".." or re.match(r"^[.\s]+$", line):
        i += 1; continue
    if not re.match(r"^[\d,]+$", line):
        name = clean_name(line)
        vals, j = [], i + 1
        while j < end and len(vals) < 4:
            v = lines[j]
            if v == "..":                       vals.append(0);           j += 1
            elif re.match(r"^[\d,]+$", v):     vals.append(parse_num(v)); j += 1
            else: break
        if len(vals) == 4:
            rows.append({
                "level":    "county" if "Embu..…" in lines[i] else "sub-county",
                "name":     name,
                "male":     vals[0],
                "female":   vals[1],
                "intersex": vals[2],
                "total":    vals[3],
            })
            i = j; continue
    i += 1

t25 = pd.DataFrame(rows)
t25.to_csv(os.path.join(OUT_DIR, "embu_table25_sex_by_subcounty.csv"), index=False)
print(t25.to_string(index=False))
print(f"  -> Saved embu_table25_sex_by_subcounty.csv\n")


# ══════════════════════════════════════════════════════════════════════════════
# VOLUME I — TABLE 2.6
# Population, Households, Average Household Size by Sub-County
# PDF page 33 (0-indexed 32)
# Columns: level | name | population | households | avg_hh_size
# ══════════════════════════════════════════════════════════════════════════════
print("Extracting Table 2.6 — Households & Avg HH Size...")

lines = get_lines(doc_vol1, 32)
start, end = find_embu_block(lines)

rows = []
i = start
while i < end:
    line = lines[i]
    if not re.match(r"^[\d,.]+$", line) and line != "..":
        name = clean_name(line)
        vals, j = [], i + 1
        while j < end and len(vals) < 3:
            v = lines[j]
            if re.match(r"^[\d,]+$", v):      vals.append(parse_num(v));   j += 1
            elif re.match(r"^\d+\.\d+$", v):  vals.append(parse_float(v)); j += 1
            else: break
        if len(vals) == 3:
            rows.append({
                "level":       "county" if "Embu..…" in lines[i] else "sub-county",
                "name":        name,
                "population":  vals[0],
                "households":  vals[1],
                "avg_hh_size": vals[2],
            })
            i = j; continue
    i += 1

t26 = pd.DataFrame(rows)
t26.to_csv(os.path.join(OUT_DIR, "embu_table26_households_by_subcounty.csv"), index=False)
print(t26.to_string(index=False))
print(f"  -> Saved embu_table26_households_by_subcounty.csv\n")


# ══════════════════════════════════════════════════════════════════════════════
# VOLUME I — TABLE 2.7
# Population, Land Area, Population Density by Sub-County
# PDF page 42 (0-indexed 41)
# Columns: level | name | population | land_area_km2 | density_per_km2
# ══════════════════════════════════════════════════════════════════════════════
print("Extracting Table 2.7 — Land Area & Population Density...")

lines = get_lines(doc_vol1, 41)
start, end = find_embu_block(lines)

rows = []
i = start
while i < end:
    line = lines[i]
    if not re.match(r"^[\d,.]+$", line) and line != "..":
        name = clean_name(line)
        vals, j = [], i + 1
        while j < end and len(vals) < 3:
            v = lines[j]
            if re.match(r"^\d[\d,]*\.\d+$", v):  vals.append(parse_float(v)); j += 1
            elif re.match(r"^\d[\d,]*$", v):      vals.append(parse_num(v));   j += 1
            else: break
        if len(vals) == 3:
            rows.append({
                "level":           "county" if "Embu..…" in lines[i] else "sub-county",
                "name":            name,
                "population":      vals[0],
                "land_area_km2":   vals[1],
                "density_per_km2": vals[2],
            })
            i = j; continue
    i += 1

t27 = pd.DataFrame(rows)
t27.to_csv(os.path.join(OUT_DIR, "embu_table27_density_by_subcounty.csv"), index=False)
print(t27.to_string(index=False))
print(f"  -> Saved embu_table27_density_by_subcounty.csv\n")


# ══════════════════════════════════════════════════════════════════════════════
# CIDP — TABLE 5: Population Projections by Age Cohort
# PDF page 45 (0-indexed 44)
# 17 age cohorts (0-4 to 80+) + Age NS + All Ages
# Columns: age_cohort | year | male | female | total
# ══════════════════════════════════════════════════════════════════════════════
print("Extracting CIDP Table 5 — Age Cohort Projections...")

doc_cidp = fitz.open(CIDP_PDF)
lines = get_lines(doc_cidp, 44)

rows = []
i = 0
while i < len(lines):
    line = lines[i]
    # Age cohort labels: "0-4", "5-9", ..., "80+", "Age NS", "All Ages"
    if (re.match(r"^\s*\d+[-+]\d*\s*$", line) or line in ("Age NS", "All Ages")):
        nums, j = [], i + 1
        while j < len(lines) and len(nums) < 12:
            v = lines[j]
            if re.match(r"^[\d,]+$", v):  nums.append(parse_num(v)); j += 1
            elif v == "-":                 nums.append(None);         j += 1
            else: break
        if len(nums) == 12:
            for yr_idx, year in enumerate(YEARS):
                o = yr_idx * 3
                rows.append({
                    "age_cohort": line.strip(),
                    "year":       year,
                    "male":       nums[o],
                    "female":     nums[o + 1],
                    "total":      nums[o + 2],
                })
            i = j; continue
    i += 1

t5 = pd.DataFrame(rows)
t5.to_csv(os.path.join(OUT_DIR, "embu_cidp_age_cohort_projections.csv"), index=False)
print(t5[t5["year"] == 2019][["age_cohort", "male", "female", "total"]].to_string(index=False))
print(f"  -> Saved embu_cidp_age_cohort_projections.csv\n")


# ══════════════════════════════════════════════════════════════════════════════
# CIDP — TABLE 6: Population Projections by Urban Area
# PDF page 46 (0-indexed 45)
# Columns: urban_area | year | male | female | total
# ══════════════════════════════════════════════════════════════════════════════
print("Extracting CIDP Table 6 — Urban Area Projections...")

lines = get_lines(doc_cidp, 45)

# Whitelist of urban centre names as they appear in the PDF
URBAN = {
    "Embu", "Runyenjes", "Siakago", "kiritiri", "Ishiara",
    "Kianjokoma", "manyattta", "Makutano", "Kibugu", "Gategi",
}

rows = []
i = 0
while i < len(lines):
    line = lines[i]
    if line in URBAN:
        nums, j = [], i + 1
        while j < len(lines) and len(nums) < 12:
            v = lines[j]
            if re.match(r"^[\d,]+$", v): nums.append(parse_num(v)); j += 1
            else: break
        if len(nums) >= 12:
            for yr_idx, year in enumerate(YEARS):
                o = yr_idx * 3
                rows.append({
                    "urban_area": line,
                    "year":       year,
                    "male":       nums[o],
                    "female":     nums[o + 1],
                    "total":      nums[o + 2],
                })
            i = j; continue
    i += 1

t6 = pd.DataFrame(rows)
t6.to_csv(os.path.join(OUT_DIR, "embu_cidp_urban_area_projections.csv"), index=False)
print(t6[t6["year"] == 2019][["urban_area", "male", "female", "total"]].to_string(index=False))
print(f"  -> Saved embu_cidp_urban_area_projections.csv\n")


# ══════════════════════════════════════════════════════════════════════════════
# CIDP — TABLE 7c: Population Distribution and Density by Ward
# PDF page 49 (0-indexed 48)
# Columns: ward | area_km2 | year | population | density
# ══════════════════════════════════════════════════════════════════════════════
print("Extracting CIDP Table 7c — Ward Population & Density...")

lines_49 = get_lines(doc_cidp, 48)

# Hard stop: Table 7c ends where Table 8 begins
table7c_end = next(i for i, l in enumerate(lines_49) if "1.5.3" in l or "Table 8" in l)

# Whitelist of ward names exactly as they appear in the PDF
WARD_NAMES = {
    "Ruguru", "Ngandori", "Kithimu", "Nginda", "Mbeti North", "Kirimari",
    "Gaturi South", "Gaturi North", "Kagaari South", "Central", "Kagaari North",
    "Kyeni North", "Kyeni South", "Mwea", "Makima", "Mbeti South", "Mavuria",
    "Kiambere", "Nthawa", "Muminji", "Evurore", "Mt. Kenya Forest",
}

rows = []
i = 0
while i < table7c_end:
    line = lines_49[i]
    if line in WARD_NAMES:
        nums, j = [], i + 1
        while j < table7c_end and len(nums) < 9:
            v = lines_49[j]
            if re.match(r"^\d[\d,.]*$", v):
                nums.append(parse_float(v) if "." in v else parse_num(v)); j += 1
            else: break
        if len(nums) == 9:
            for yr_idx, year in enumerate(YEARS):
                o = 1 + yr_idx * 2
                rows.append({
                    "ward":       line,
                    "area_km2":   nums[0],
                    "year":       year,
                    "population": nums[o],
                    "density":    nums[o + 1],
                })
            i = j; continue
    # Handle "Mt. Kenya Forest" split across two lines
    elif line == "Mt. Kenya" and i + 1 < table7c_end and lines_49[i + 1] == "Forest":
        nums, j = [], i + 2
        while j < table7c_end and len(nums) < 9:
            v = lines_49[j]
            if re.match(r"^\d[\d,.]*$", v):
                nums.append(parse_float(v) if "." in v else parse_num(v)); j += 1
            else: break
        if len(nums) == 9:
            for yr_idx, year in enumerate(YEARS):
                o = 1 + yr_idx * 2
                rows.append({
                    "ward":       "Mt. Kenya Forest",
                    "area_km2":   nums[0],
                    "year":       year,
                    "population": nums[o],
                    "density":    nums[o + 1],
                })
            i = j; continue
    i += 1

t7c = pd.DataFrame(rows).drop_duplicates(["ward", "year"])
t7c.to_csv(os.path.join(OUT_DIR, "embu_cidp_ward_density_projections.csv"), index=False)
print(f"  Wards: {t7c['ward'].nunique()}")
print(t7c[t7c["year"] == 2019][["ward", "area_km2", "population", "density"]].to_string(index=False))
print(f"  -> Saved embu_cidp_ward_density_projections.csv\n")


# ══════════════════════════════════════════════════════════════════════════════
# CIDP — TABLE 8: Population Projections by Broad Age Groups
# PDF pages 49-50 (0-indexed 48-49)
# Labels are multi-line in the PDF — matched by unique trigger keyword
# Columns: age_group | year | male | female | total
# ══════════════════════════════════════════════════════════════════════════════
print("Extracting CIDP Table 8 — Broad Age Group Projections...")

# Combine Table 8 section of page 49 with all of page 50
lines_t8 = lines_49[table7c_end:] + get_lines(doc_cidp, 49)

# Each tuple: (trigger keyword found in the label line, clean output label)
BROAD_GROUPS = [
    ("Infant",                "infant_under1"),
    ("Under 5",               "under5"),
    ("Pre-School",            "preschool_3to5"),
    ("Primary",               "primary_6to13"),
    ("Secondary",             "secondary_13to19"),
    ("Youth",                 "youth_15to29"),
    ("Women of Reproductive", "women_reproductive_15to49"),
    ("Economically",          "economically_active_15to64"),
    ("Aged",                  "aged_65plus"),
]

rows = []
seen = set()

for trigger, label in BROAD_GROUPS:
    if label in seen:
        continue
    # Find trigger in the combined lines
    idx = next((i for i, l in enumerate(lines_t8) if trigger in l), None)
    if idx is None:
        print(f"  WARNING: trigger '{trigger}' not found — skipping {label}")
        continue
    # Skip non-numeric label fragments until we collect 12 numbers
    nums, j = [], idx + 1
    while j < len(lines_t8) and len(nums) < 12:
        v = lines_t8[j]
        if re.match(r"^[\d,]+$", v):  nums.append(parse_num(v)); j += 1
        elif v == "-":                 nums.append(None);         j += 1
        else:                          j += 1  # skip label continuation lines
        if len(nums) == 12:
            break
    if len(nums) == 12:
        seen.add(label)
        for yr_idx, year in enumerate(YEARS):
            o = yr_idx * 3
            rows.append({
                "age_group": label,
                "year":      year,
                "male":      nums[o],
                "female":    nums[o + 1],
                "total":     nums[o + 2],
            })
    else:
        print(f"  WARNING: only found {len(nums)} numbers for {label} — skipping")

t8 = pd.DataFrame(rows)
t8.to_csv(os.path.join(OUT_DIR, "embu_cidp_broad_age_projections.csv"), index=False)
print(t8[t8["year"] == 2019][["age_group", "male", "female", "total"]].to_string(index=False))
print(f"  -> Saved embu_cidp_broad_age_projections.csv\n")


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("VALIDATION")
print("=" * 60)

pop_t25  = t25[t25["level"] == "county"]["total"].values[0]
pop_t27  = t27[t27["level"] == "county"]["population"].values[0]
sub_sum  = t25[t25["level"] == "sub-county"]["total"].sum()
all_ages = t5[t5["age_cohort"] == "All Ages"]
all_ages_19 = all_ages[all_ages["year"] == 2019]["total"].values[0] if len(all_ages) else "not found"

print(f"Table 2.5 county total:     {pop_t25:,}  {'✓' if pop_t25 == 608_599 else '✗ expected 608,599'}")
print(f"Table 2.7 county total:     {pop_t27:,}  {'✓' if pop_t27 == 608_599 else '✗ expected 608,599'}")
print(f"Sub-county totals sum:      {sub_sum:,}  {'✓' if sub_sum == pop_t25 else '✗ mismatch'}")
print(f"CIDP All Ages 2019:         {all_ages_19}  (expected ~608,575)")
print(f"Age cohorts extracted:      {t5['age_cohort'].nunique()}  (expected 17)")
print(f"Urban areas extracted:      {t6['urban_area'].nunique()}  (expected 10)")
print(f"Wards extracted:            {t7c['ward'].nunique()}  (expected 21)")
print(f"Broad age groups extracted: {t8['age_group'].nunique()}  (expected 9)")
print(f"\nFiles saved to: {OUT_DIR}")