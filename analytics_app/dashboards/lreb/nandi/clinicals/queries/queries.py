"""
queries.py: Nandi County OPD Clinical Dashboard.

All Snowflake queries for HOSPITALS.NANDI_CLEAN.NANDI_OUTPATIENT_WIDE.
Connection pattern mirrors the SPH dashboard:
    from snowflake_service.snowflake_client import SnowflakeClient

Rules
- Every public function is decorated @st.cache_data(ttl=3600, show_spinner=False), so no function names flash on screen while a tab loads.
- _run() is the only place SQL touches the connection. Zero SQL in views.
- (is_default_code_pair = FALSE OR is_default_code_pair IS NULL) excludes the AAA
  default code pair from every disease query.
- Disease categories overlap by design (pneumonia is both Infectious and Respiratory,
  malaria sits inside Infectious). Never sum across categories.
- Comparable full years are 2024 and 2025. 2023 and 2026 are partial.
- Set NANDI_DEMO=1 to serve illustrative data from demo_data.py (no Snowflake needed).

Tab 2 (Population & Geography) queries are the q2_* functions.
Tab 3 (Infectious disease) queries are the q3_* functions (Q3.x from NANDI QUERIES.txt, N3.x new for the dashboard).
Tab 4 (NCDs) queries are the q4_* functions (Q4.x from ncd_queries_v3.sql, N4.x new for the dashboard).
"""
from __future__ import annotations

import os
from decimal import Decimal

import pandas as pd
import streamlit as st

TABLE = "HOSPITALS.NANDI_CLEAN.NANDI_OUTPATIENT_WIDE"
GEO_TABLE = "HOSPITALS.LOOKUP_TABLES.NANDI_GEO_COORDINATES"
DEMO = os.getenv("NANDI_DEMO", "").strip() in ("1", "true", "True", "yes")

# !! CONFIRM with the hospital. Placeholder is approximate (Kapsabet town centre).
HOSPITAL_LAT = 0.2030
HOSPITAL_LON = 35.1050

INTERNATIONAL = ("Rwanda", "United Kingdom", "Uganda")
_INTL_SQL = ", ".join(f"'{c}'" for c in INTERNATIONAL)
NOT_DEFAULT_PAIR = "(is_default_code_pair = FALSE OR is_default_code_pair IS NULL)"
# Tindiret and Tinderet are one sub-county spelled two ways; the view merges them.

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        from snowflake_service.snowflake_client import SnowflakeClient

        _CLIENT = SnowflakeClient()
    return _CLIENT


def _demo():
    """Load demo_data.py from this folder (works however queries.py itself was imported)."""
    import importlib.util
    import sys

    mod = sys.modules.get("nandi_demo_data")
    if mod is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data.py")
        spec = importlib.util.spec_from_file_location("nandi_demo_data", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["nandi_demo_data"] = mod
        spec.loader.exec_module(mod)
    return mod


def _decimals_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """Snowflake returns NUMBER results as decimal.Decimal, which can't mix with float in pandas/Python math."""
    for c in df.columns:  # checked by value, not dtype: pyarrow decimal columns aren't dtype object
        if df[c].map(lambda v: isinstance(v, Decimal)).any():
            df[c] = pd.to_numeric(df[c].map(lambda v: float(v) if isinstance(v, Decimal) else v), errors="coerce")
    return df


def _run(name: str, sql: str) -> pd.DataFrame:
    """The only place SQL touches the connection. Returns lower-case column names."""
    try:
        if DEMO:
            return _demo().get(name)
        df = _client().query(sql)
        df.columns = [c.lower() for c in df.columns]
        return _decimals_to_float(df)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Query error ({name}): {exc}")
        return pd.DataFrame()


# --------------------------------------------------------------- shared SQL pieces
CATEGORY_RULES: dict[str, list[str]] = {
    "Infectious disease": ["pneumonia", "sepsis", "malaria", "gastritis", "urinary tract infection"],
    "Respiratory": ["pneumonia", "respiratory", "asthma", "tonsil", "bronch"],
    "NCD": ["hypertension", "diabetes", "chronic kidney", "epilepsy"],
    "Injury & trauma": ["fracture", "wound", "burn", "injury"],
    "Maternal & obstetric": ["pregnan", "obstetric", "abortion", "puerperi"],
    "Mental health": ["schizophrenia", "alcohol", "bipolar", "depression"],
    "Malaria": ["malaria", "plasmodium"],
}


def _cats_cte(with_all: bool = False) -> str:
    """CTE `cats(category, pattern)`: one row per disease category with a regex of its keywords.
    Visits join to it on all_diagnoses (every diagnosis of the visit, concatenated), so a visit
    counts once per category it matches and no array / FLATTEN is needed.
    Snowflake's REGEXP_LIKE must match the whole string, hence the .* either side."""
    rows = [(n, ".*(" + "|".join(p) + ").*") for n, p in CATEGORY_RULES.items()]
    if with_all:
        rows.append(("All visits", ".*"))
    vals = ",\n        ".join(f"('{n}', '{p}')" for n, p in rows)
    return f"cats(category, pattern) AS (\n    SELECT column1, column2 FROM VALUES\n        {vals})"


_DX_MATCH = "REGEXP_LIKE(COALESCE(w.all_diagnoses, ''), c.pattern, 'is')"


_AGE_CASE = """
    CASE
        WHEN age_in_years_float < 5  THEN 'Toddler (0-4)'
        WHEN age_in_years_float < 13 THEN 'Child (5-12)'
        WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
        WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
        WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
        WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
        WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
        ELSE '55+'
    END"""

# Q2.6 reason flags (diagnosis_1 only), reused for every county
_INJURY = """(diagnosis_1 ILIKE '%fracture%' OR diagnosis_1 ILIKE '%injury%' OR diagnosis_1 ILIKE '%wound%'
              OR diagnosis_1 ILIKE '%road traffic%' OR diagnosis_1 ILIKE '%transport%' OR diagnosis_1 ILIKE '%burn%')"""
_ACUTE = "(diagnosis_1 ILIKE '%pneumonia%' OR diagnosis_1 ILIKE '%sepsis%' OR diagnosis_1 ILIKE '%malaria%')"
_NCD = "(diagnosis_1 ILIKE '%hypertension%' OR diagnosis_1 ILIKE '%diabetes%' OR diagnosis_1 ILIKE '%chronic kidney%')"


# =========================================================================== TAB 1 (shared with tab 2)
@st.cache_data(ttl=3600, show_spinner=False)
def q1_headline_kpis() -> pd.DataFrame:
    """Q1.1 Total visits, unique patients, % female (the baseline every tab compares against), % under-5."""
    return _run("q1_headline_kpis", f"""
    SELECT COUNT(*)                                             AS total_visits,
           COUNT(DISTINCT unique_patient_id)                    AS unique_patients,
           MIN(visit_year)                                      AS earliest_year,
           MAX(visit_year)                                      AS latest_year,
           ROUND(AVG(CASE WHEN sex = 'F' THEN 1.0 ELSE 0.0 END) * 100, 1)             AS pct_female,
           ROUND(AVG(CASE WHEN age_in_years_float < 5 THEN 1.0 ELSE 0.0 END) * 100, 1) AS pct_under5
    FROM {TABLE}
    WHERE {NOT_DEFAULT_PAIR}
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q1_disease_category_summary() -> pd.DataFrame:
    """Q1.5 Visits per disease category (categories overlap; never sum them)."""
    return _run("q1_disease_category_summary", f"""
    WITH {_cats_cte()},
    w AS (SELECT all_diagnoses FROM {TABLE} WHERE {NOT_DEFAULT_PAIR})
    SELECT c.category, COUNT(*) AS visits
    FROM w JOIN cats c ON {_DX_MATCH}
    GROUP BY c.category ORDER BY visits DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q1_age_sex() -> pd.DataFrame:
    """Q1.3 Visits by age group and sex."""
    return _run("q1_age_sex", f"""
    SELECT {_AGE_CASE} AS age_group,
           SUM(CASE WHEN sex = 'F' THEN 1 ELSE 0 END) AS female,
           SUM(CASE WHEN sex = 'M' THEN 1 ELSE 0 END) AS male
    FROM {TABLE}
    WHERE age_in_years_float IS NOT NULL
    GROUP BY age_group
    ORDER BY MIN(age_in_years_float)
    """)


# =========================================================================== TAB 2 · Population & Geography
@st.cache_data(ttl=3600, show_spinner=False)
def q2_origin() -> pd.DataFrame:
    """Q2.5 Nandi vs other Kenyan county vs international vs unmatched."""
    return _run("q2_origin", f"""
    SELECT CASE
             WHEN county = 'Nandi'                    THEN 'Nandi County'
             WHEN county IN ({_INTL_SQL})             THEN 'International'
             WHEN county IS NOT NULL AND county != 'Nandi' THEN 'Other Kenyan county'
             WHEN geo_confidence = 'NOT_A_PLACE'      THEN 'Noise / unresolvable'
             ELSE 'Residence unmatched'
           END AS patient_origin,
           COUNT(*)                                           AS visits,
           COUNT(DISTINCT unique_patient_id)                  AS unique_patients,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total,
           SUM(CASE WHEN visit_year = 2024 THEN 1 ELSE 0 END) AS visits_2024,
           SUM(CASE WHEN visit_year = 2025 THEN 1 ELSE 0 END) AS visits_2025
    FROM {TABLE}
    GROUP BY patient_origin
    ORDER BY visits DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q2_place_summary() -> pd.DataFrame:
    """N1a Visits and patients at county, sub-county and ward level in one pass (all counties)."""
    return _run("q2_place_summary", f"""
    SELECT CASE GROUPING_ID(county, sub_county, ward) WHEN 3 THEN 'county' WHEN 1 THEN 'sub' ELSE 'ward' END AS level,
           county, sub_county, ward,
           COUNT(*)                                           AS visits,
           COUNT(DISTINCT unique_patient_id)                  AS unique_patients,
           SUM(IFF(visit_year = 2024, 1, 0))                  AS visits_2024,
           SUM(IFF(visit_year = 2025, 1, 0))                  AS visits_2025,
           ROUND(SUM(IFF(sex = 'F', 1.0, 0.0)) * 100.0 / NULLIF(COUNT(*), 0), 1) AS pct_female
    FROM {TABLE}
    WHERE county IS NOT NULL AND county NOT IN ({_INTL_SQL}) 
    GROUP BY GROUPING SETS ((county), (county, sub_county), (county, sub_county, ward))
    HAVING GROUPING_ID(county, sub_county, ward) = 3
        OR (GROUPING_ID(county, sub_county, ward) = 1 AND sub_county IS NOT NULL)
        OR (GROUPING_ID(county, sub_county, ward) = 0 AND sub_county IS NOT NULL AND ward IS NOT NULL)
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q2_county_profile() -> pd.DataFrame:
    """Q2.6 (extended to every county, Nandi included): volume, age, sex, reason for visit, by year."""
    return _run("q2_county_profile", f"""
    SELECT county,
           COUNT(*)                                  AS visits,
           COUNT(DISTINCT unique_patient_id)         AS patients,
           SUM(IFF(visit_year = 2024, 1, 0))         AS visits_2024,
           SUM(IFF(visit_year = 2025, 1, 0))         AS visits_2025,
           ROUND(AVG(age_in_years_float), 1)         AS mean_age,
           ROUND(SUM(IFF(sex = 'F', 1.0, 0.0)) * 100.0 / NULLIF(COUNT(*), 0), 1) AS pct_female,
           SUM(IFF({_INJURY}, 1, 0))                 AS injury_visits,
           SUM(IFF({_ACUTE}, 1, 0))                  AS acute_illness_visits,
           SUM(IFF({_NCD}, 1, 0))                    AS ncd_visits
    FROM {TABLE}
    WHERE county IS NOT NULL AND county NOT IN ({_INTL_SQL}) 
    GROUP BY county
    ORDER BY visits DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q2_category_by_place() -> pd.DataFrame:
    """N3 Disease category x county / sub-county / ward (2024 + 2025). Categories overlap by design."""
    return _run("q2_category_by_place", f"""
    WITH {_cats_cte()},
    w AS (
        SELECT county,
               COALESCE(sub_county, '(none)') AS sub_county,
               COALESCE(ward, '(none)')       AS ward,
               all_diagnoses,
               COUNT(*) OVER (PARTITION BY county, COALESCE(sub_county, '(none)'), COALESCE(ward, '(none)')) AS area_visits
        FROM {TABLE}
        WHERE {NOT_DEFAULT_PAIR} AND county IS NOT NULL AND county NOT IN ({_INTL_SQL})
           AND visit_year IN (2024, 2025)
    )
    SELECT w.county, w.sub_county, w.ward, c.category,
           COUNT(*) AS category_visits, ANY_VALUE(w.area_visits) AS area_visits
    FROM w JOIN cats c ON {_DX_MATCH}
    GROUP BY w.county, w.sub_county, w.ward, c.category
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q2_locality_distance() -> pd.DataFrame:
    """N2 / N2b Visits per locality and disease category with km from the hospital (HAVERSINE)."""
    return _run("q2_locality_distance", f"""
    WITH {_cats_cte(with_all=True)},
    loc_geo AS (
        SELECT UPPER(TRIM(geo_name)) AS k, ANY_VALUE(latitude) AS lat, ANY_VALUE(longitude) AS lon
        FROM {GEO_TABLE}
        WHERE geo_level = 'LOCALITY' AND geocode_status = 'FOUND'
          AND latitude BETWEEN -5 AND 5.5 AND longitude BETWEEN 33.5 AND 42   -- reject geocoder hits outside Kenya
        GROUP BY 1
    ),
    w AS (
        SELECT locality, ward, sub_county, county, all_diagnoses
        FROM {TABLE}
        WHERE locality IS NOT NULL  AND {NOT_DEFAULT_PAIR}
    )
    SELECT w.locality, ANY_VALUE(w.ward) AS ward, ANY_VALUE(w.sub_county) AS sub_county, ANY_VALUE(w.county) AS county,
           g.lat AS latitude, g.lon AS longitude,
           ROUND(HAVERSINE({HOSPITAL_LAT}, {HOSPITAL_LON}, g.lat, g.lon), 2) AS km,
           c.category, COUNT(*) AS visits
    FROM w
    JOIN loc_geo g ON g.k = UPPER(TRIM(w.locality))
    JOIN cats c ON {_DX_MATCH}
    GROUP BY w.locality, g.lat, g.lon, c.category
    HAVING COUNT(*) >= 5
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q2_other_county_distance() -> pd.DataFrame:
    """N2d Other counties: approximate distance from the hospital (county centre) with reason mix."""
    return _run("q2_other_county_distance", f"""
    WITH cty AS (
        SELECT county,
               COUNT(*) AS visits, COUNT(DISTINCT unique_patient_id) AS unique_patients,
               SUM(IFF({_INJURY}, 1, 0)) AS injury_visits,
               SUM(IFF({_ACUTE}, 1, 0))  AS acute_illness_visits,
               SUM(IFF({_NCD}, 1, 0))    AS ncd_visits
        FROM {TABLE}
        WHERE county IS NOT NULL AND county <> 'Nandi' AND county NOT IN ({_INTL_SQL}) 
        GROUP BY county
    )
    SELECT c.county, c.visits, c.unique_patients, c.injury_visits, c.acute_illness_visits, c.ncd_visits,
           ROUND(HAVERSINE({HOSPITAL_LAT}, {HOSPITAL_LON}, g.latitude, g.longitude), 0) AS km_from_hospital
    FROM cty c
    LEFT JOIN {GEO_TABLE} g
           ON g.geo_level = 'COUNTY' AND UPPER(TRIM(g.geo_name)) = UPPER(TRIM(c.county)) AND g.geocode_status = 'FOUND'
    ORDER BY km_from_hospital
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q2_locality_growth() -> pd.DataFrame:
    """N6 Growth watch: locality visits 2024 vs 2025 with share of its sub-county and new patients."""
    return _run("q2_locality_growth", f"""
    WITH loc_year AS (
        SELECT sub_county, locality, visit_year, COUNT(*) AS visits, COUNT(DISTINCT unique_patient_id) AS patients
        FROM {TABLE}
        WHERE locality IS NOT NULL  AND visit_year IN (2024, 2025)
        GROUP BY sub_county, locality, visit_year
    ),
    first_seen AS (SELECT unique_patient_id, MIN(visit_year) AS first_year FROM {TABLE} GROUP BY 1),
    new_pat AS (
        SELECT w.locality, w.visit_year, COUNT(DISTINCT w.unique_patient_id) AS new_patients
        FROM {TABLE} w
        JOIN first_seen f ON f.unique_patient_id = w.unique_patient_id AND f.first_year = w.visit_year
        WHERE w.visit_year IN (2024, 2025) AND w.locality IS NOT NULL
        GROUP BY w.locality, w.visit_year
    ),
    wide AS (
        SELECT sub_county, locality,
               SUM(IFF(visit_year = 2024, visits, 0))   AS visits_2024,
               SUM(IFF(visit_year = 2025, visits, 0))   AS visits_2025,
               SUM(IFF(visit_year = 2024, patients, 0)) AS patients_2024,
               SUM(IFF(visit_year = 2025, patients, 0)) AS patients_2025
        FROM loc_year GROUP BY sub_county, locality
    )
    SELECT w.sub_county, w.locality, w.visits_2024, w.visits_2025, w.patients_2024, w.patients_2025,
           w.visits_2024 * 100.0 / NULLIF(SUM(w.visits_2024) OVER (PARTITION BY w.sub_county), 0) AS share_sub_2024,
           w.visits_2025 * 100.0 / NULLIF(SUM(w.visits_2025) OVER (PARTITION BY w.sub_county), 0) AS share_sub_2025,
           COALESCE(n24.new_patients, 0) AS new_patients_2024, COALESCE(n25.new_patients, 0) AS new_patients_2025
    FROM wide w
    LEFT JOIN new_pat n24 ON n24.locality = w.locality AND n24.visit_year = 2024
    LEFT JOIN new_pat n25 ON n25.locality = w.locality AND n25.visit_year = 2025
    WHERE w.visits_2024 + w.visits_2025 >= 50
    ORDER BY w.visits_2024 + w.visits_2025 DESC
    LIMIT 40
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q2_boundaries() -> pd.DataFrame:
    """N1b Sub-county and ward polygons (BOUNDARY_GEOJSON) for every county. Loaded once and cached."""
    return _run("q2_boundaries", f"""
    SELECT geo_level, geo_name, parent_name, county, latitude, longitude, boundary_geojson
    FROM {GEO_TABLE}
    WHERE geo_level IN ('SUB_COUNTY', 'WARD') AND geocode_status = 'FOUND' AND boundary_geojson IS NOT NULL
    """)


# =========================================================================== TAB 3 · Infectious disease
# Matching follows the Tab 3 notes: ALL_DIAGNOSES ILIKE, and the diagnosis_1..8 FLATTEN pattern where single
# diagnoses are needed. The default code-pair exclusion is applied only where the original Q3.x query applied it.
# Q3.x come from NANDI QUERIES.txt; N3.x are new for this dashboard.

@st.cache_data(ttl=3600, show_spinner=False)
def q3_infectious_patients() -> pd.DataFrame:
    """N3.0 KPI: patients and visits with any infectious diagnosis, by year plus an all-years row."""
    return _run("q3_infectious_patients", f"""
    WITH base AS (
        SELECT visit_year, unique_patient_id,
               IFF(ALL_DIAGNOSES ILIKE ANY ('%pneumonia%','%sepsis%','%malaria%','%plasmodium%',
            '%urinary tract infection%','%gastritis%','%tonsil%',
            '%gastroenteritis%','%tuberculosis%','%gonococcal%',
            '%syphilis%','%chlamyd%','%typhoid%','%hepatitis%'), 1, 0) AS is_infectious
        FROM {TABLE}
        WHERE unique_patient_id IS NOT NULL AND ALL_DIAGNOSES IS NOT NULL
    )
    SELECT
        COALESCE(TO_VARCHAR(visit_year), 'All years')              AS period,
        COUNT(DISTINCT unique_patient_id)                          AS all_patients,
        COUNT(DISTINCT IFF(is_infectious = 1, unique_patient_id, NULL)) AS infectious_patients,
        ROUND(COUNT(DISTINCT IFF(is_infectious = 1, unique_patient_id, NULL))
              * 100.0 / NULLIF(COUNT(DISTINCT unique_patient_id), 0), 1) AS pct_of_patients,
        COUNT(*)                                                   AS all_visits,
        SUM(is_infectious)                                         AS infectious_visits
    FROM base
    GROUP BY GROUPING SETS ((visit_year), ())
    ORDER BY period
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_malaria_species_year() -> pd.DataFrame:
    """Q3.1 Malaria species by year (falciparum, malariae, vivax, artemisinin-resistant)."""
    return _run("q3_malaria_species_year", f"""
    WITH dx_long AS (
        SELECT pno, visit_year, sex, age_in_years_float, VALUE::STRING AS VALUE
        FROM {TABLE},
             LATERAL FLATTEN(INPUT => ARRAY_CONSTRUCT(
                 diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4,
                 diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
             )) f
        WHERE VALUE ILIKE '%malaria%' OR VALUE ILIKE '%plasmodium%'
    )
    SELECT
        visit_year,
        CASE
            WHEN VALUE ILIKE '%falciparum%'  THEN 'P. falciparum'
            WHEN VALUE ILIKE '%malariae%'    THEN 'P. malariae'
            WHEN VALUE ILIKE '%vivax%'       THEN 'P. vivax'
            WHEN VALUE ILIKE '%artemisinin%' THEN 'Artemisinin-resistant Pf'
            ELSE 'Unspecified'
        END                                        AS species,
        COUNT(DISTINCT pno)                        AS visits,
        ROUND(AVG(age_in_years_float), 1)         AS mean_age,
        SUM(CASE WHEN sex = 'F' THEN 1 ELSE 0 END) AS female,
        SUM(CASE WHEN sex = 'M' THEN 1 ELSE 0 END) AS male
    FROM dx_long
    GROUP BY visit_year, species
    ORDER BY visit_year, visits DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_malaria_age() -> pd.DataFrame:
    """Q3.2 Malaria species by age group."""
    return _run("q3_malaria_age", f"""
    WITH dx_long AS (
        SELECT pno, visit_year, sex, age_in_years_float, VALUE::STRING AS VALUE
        FROM {TABLE},
             LATERAL FLATTEN(INPUT => ARRAY_CONSTRUCT(
                 diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4,
                 diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
             )) f
        WHERE VALUE ILIKE '%malaria%' OR VALUE ILIKE '%plasmodium%'
    )
    SELECT
        CASE
            WHEN age_in_years_float < 5  THEN 'Toddler (0-4)'
            WHEN age_in_years_float < 13 THEN 'Child (5-12)'
            WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
            WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
            WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
            WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
            WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
            ELSE '55+'
        END                                        AS age_group,
        SUM(CASE WHEN VALUE ILIKE '%falciparum%'  THEN 1 ELSE 0 END) AS p_falciparum,
        SUM(CASE WHEN VALUE ILIKE '%malariae%'    THEN 1 ELSE 0 END) AS p_malariae,
        SUM(CASE WHEN VALUE ILIKE '%vivax%'       THEN 1 ELSE 0 END) AS p_vivax,
        SUM(CASE WHEN VALUE ILIKE '%artemisinin%' THEN 1 ELSE 0 END) AS artemisinin_resistant,
        SUM(CASE WHEN VALUE NOT ILIKE '%falciparum%'
                 AND VALUE NOT ILIKE '%malariae%'
                 AND VALUE NOT ILIKE '%vivax%'
                 AND VALUE NOT ILIKE '%artemisinin%' THEN 1 ELSE 0 END) AS unspecified,
        COUNT(DISTINCT pno)                        AS total_malaria_visits,
        SUM(CASE WHEN sex = 'F' THEN 1 ELSE 0 END) AS female,
        SUM(CASE WHEN sex = 'M' THEN 1 ELSE 0 END) AS male,
        ROUND(AVG(age_in_years_float), 1)         AS mean_age
    FROM dx_long
    WHERE age_in_years_float IS NOT NULL
    GROUP BY age_group
    ORDER BY MIN(age_in_years_float)
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_malaria_rain() -> pd.DataFrame:
    """Q3.2b Malaria per 1,000 OPD visits against Eldoret annual rainfall (NOAA)."""
    return _run("q3_malaria_rain", f"""
    WITH malaria_annual AS (
        SELECT
            visit_year,
            SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%malaria%'
                     OR ALL_DIAGNOSES ILIKE '%plasmodium%' THEN 1 ELSE 0 END) AS malaria_visits,
            SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%falciparum%' THEN 1 ELSE 0 END) AS pf_visits,
            SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%malariae%'   THEN 1 ELSE 0 END) AS pm_visits,
            COUNT(*)                                                            AS total_opd_visits,
            ROUND(SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%malaria%'
                           THEN 1.0 ELSE 0.0 END)
                  / NULLIF(COUNT(*), 0) * 1000, 2)                             AS malaria_per_1000_opd
        FROM {TABLE}
        WHERE ALL_DIAGNOSES IS NOT NULL
        GROUP BY visit_year
    ),
    weather_annual AS (
        SELECT
            YEAR(date)                             AS weather_year,
            ROUND(SUM(CASE WHEN variable ILIKE '%precipitation%'
                           THEN value ELSE 0 END), 0) AS annual_rainfall_mm,
            ROUND(AVG(CASE WHEN variable ILIKE '%maximum temperature%'
                           THEN value END), 1)         AS avg_tmax_c,
            ROUND(AVG(CASE WHEN variable ILIKE '%minimum temperature%'
                           THEN value END), 1)         AS avg_tmin_c
        FROM SNOWFLAKE_PUBLIC_DATA_FREE.PUBLIC_DATA_FREE.NOAA_WEATHER_METRICS_TIMESERIES
        WHERE noaa_weather_station_id = 'KEM00063686'  -- Eldoret International
          AND YEAR(date) BETWEEN 2022 AND 2026
        GROUP BY YEAR(date)
    )
    SELECT
        m.visit_year,
        m.malaria_visits,
        m.pf_visits,
        m.pm_visits,
        m.total_opd_visits,
        m.malaria_per_1000_opd,
        COALESCE(w.annual_rainfall_mm, 0)          AS annual_rainfall_mm,
        w.avg_tmax_c,
        w.avg_tmin_c,
        CASE
            WHEN w.annual_rainfall_mm IS NULL
              OR w.annual_rainfall_mm = 0 THEN 'No weather data — station offline'
            WHEN m.visit_year IN (2023, 2026)      THEN 'Partial OPD year — exclude from correlation'
            ELSE 'Full year'
        END                                        AS data_quality_flag
    FROM malaria_annual m
    LEFT JOIN weather_annual w ON m.visit_year = w.weather_year
    ORDER BY m.visit_year
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_pneumonia_age() -> pd.DataFrame:
    """Q3.3 Pneumonia by age group and year: severity and co-diagnoses."""
    return _run("q3_pneumonia_age", f"""
    SELECT
        CASE
            WHEN age_in_years_float < 5  THEN 'Toddler (0-4)'
            WHEN age_in_years_float < 13 THEN 'Child (5-12)'
            WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
            WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
            WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
            WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
            WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
            ELSE '55+'
        END                                        AS age_group,
        visit_year,
        COUNT(*)                                   AS pneumonia_visits,
        SUM(CASE WHEN sex = 'F' THEN 1 ELSE 0 END) AS female,
        SUM(CASE WHEN sex = 'M' THEN 1 ELSE 0 END) AS male,
        SUM(CASE WHEN is_severe_problem   = TRUE THEN 1 ELSE 0 END) AS severe_problem,
        SUM(CASE WHEN is_complete_problem = TRUE THEN 1 ELSE 0 END) AS complete_problem,
        ROUND(SUM(CASE WHEN is_severe_problem = TRUE THEN 1.0 ELSE 0.0 END)
              / NULLIF(COUNT(*), 0) * 100, 1)      AS pct_severe,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%anaemia%'       THEN 1 ELSE 0 END) AS with_anaemia,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%diabetes%'      THEN 1 ELSE 0 END) AS with_dm,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%heart failure%' THEN 1 ELSE 0 END) AS with_heart_failure,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%asthma%'
              OR ALL_DIAGNOSES ILIKE '%copd%'      THEN 1 ELSE 0 END) AS with_respiratory_ncd,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%hiv%'           THEN 1 ELSE 0 END) AS with_hiv,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%malnutrition%'
              OR ALL_DIAGNOSES ILIKE '%undernutrition%'
              OR ALL_DIAGNOSES ILIKE '%wasting%'   THEN 1 ELSE 0 END) AS with_malnutrition
    FROM {TABLE}
    WHERE ALL_DIAGNOSES ILIKE '%pneumonia%'
      AND age_in_years_float IS NOT NULL
      AND visit_year IN (2024, 2025)
    GROUP BY age_group, visit_year
    ORDER BY visit_year, MIN(age_in_years_float)
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_sepsis_types() -> pd.DataFrame:
    """N3.S0 Sepsis classification by type and age group. "without septic shock" is tested before "septic shock", so each visit lands in exactly one type."""
    return _run("q3_sepsis_types", f"""
    SELECT
        CASE
            WHEN ALL_DIAGNOSES ILIKE '%without septic%'            THEN 'Sepsis without septic shock'
            WHEN ALL_DIAGNOSES ILIKE '%septic shock%'              THEN 'Septic shock'
            WHEN ALL_DIAGNOSES ILIKE '%severe sepsis%'             THEN 'Severe sepsis'
            WHEN ALL_DIAGNOSES ILIKE '%puerperal sepsis%'          THEN 'Puerperal sepsis'
            WHEN ALL_DIAGNOSES ILIKE '%neonatal sepsis%'           THEN 'Neonatal sepsis'
            WHEN ALL_DIAGNOSES ILIKE '%mild sepsis%'               THEN 'Mild sepsis'
            ELSE 'Other sepsis'
        END                                                     AS sepsis_type,
        CASE
            WHEN age_in_years_float < 5  THEN 'Toddler (0-4)'
            WHEN age_in_years_float < 13 THEN 'Child (5-12)'
            WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
            WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
            WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
            WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
            WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
            ELSE '55+'
        END                                                             AS age_group,
        COUNT(*)                                                           AS visits,
        SUM(IFF(sex = 'F', 1, 0))                                          AS female,
        SUM(IFF(sex = 'M', 1, 0))                                          AS male,
        SUM(IFF(is_severe_problem = TRUE, 1, 0))                           AS severe_flag
    FROM {TABLE}
    WHERE ALL_DIAGNOSES ILIKE '%sepsis%' AND age_in_years_float IS NOT NULL
    GROUP BY sepsis_type, age_group
    ORDER BY visits DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_toddler_sepsis_codx() -> pd.DataFrame:
    """Q3.4b Top co-diagnoses on toddler sepsis visits."""
    return _run("q3_toddler_sepsis_codx", f"""
    WITH dx_long AS (
        SELECT pno, VALUE::STRING AS dx
        FROM {TABLE},
             LATERAL FLATTEN(INPUT => ARRAY_CONSTRUCT(
                 diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4,
                 diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
             )) f
        WHERE ALL_DIAGNOSES ILIKE '%sepsis%'
          AND age_in_years_float < 5
          AND VALUE NOT ILIKE '%sepsis%'
          AND VALUE IS NOT NULL
    )
    SELECT
        dx                                         AS co_diagnosis,
        COUNT(*)                                   AS occurrences,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
    FROM dx_long
    GROUP BY dx
    ORDER BY occurrences DESC
    LIMIT 15
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_puerperal_sepsis() -> pd.DataFrame:
    """Q3.4c Puerperal sepsis by year and age group."""
    return _run("q3_puerperal_sepsis", f"""
    SELECT
        visit_year,
        CASE
            WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
            WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
            WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
            WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
            ELSE '45+'
        END                                        AS age_group,
        COUNT(*)                                   AS visits,
        SUM(CASE WHEN is_severe_problem = TRUE THEN 1 ELSE 0 END) AS severe_cases
    FROM {TABLE}
    WHERE ALL_DIAGNOSES ILIKE '%puerperal sepsis%'
    GROUP BY visit_year, age_group
    ORDER BY visit_year, visits DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_uti_sex_age() -> pd.DataFrame:
    """Q3.5 UTI by sex and age group."""
    return _run("q3_uti_sex_age", f"""
    SELECT
        sex,
        CASE
            WHEN age_in_years_float < 5  THEN 'Toddler (0-4)'
            WHEN age_in_years_float < 13 THEN 'Child (5-12)'
            WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
            WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
            WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
            WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
            WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
            ELSE '55+'
        END                                        AS age_group,
        COUNT(*)                                   AS visits,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct_of_total
    FROM {TABLE}
    WHERE ALL_DIAGNOSES ILIKE '%urinary tract infection%'
      AND age_in_years_float IS NOT NULL
      AND sex IN ('F','M')
    GROUP BY sex, age_group
    ORDER BY sex, MIN(age_in_years_float)
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_uti_female_complicated() -> pd.DataFrame:
    """Q3.5b Female UTI: uncomplicated vs complicated, by age."""
    return _run("q3_uti_female_complicated", f"""
    WITH female_uti AS (
        SELECT
            pno, age_in_years_float, ALL_DIAGNOSES,
            CASE
                WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
                WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
                WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
                WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
                ELSE '55+'
            END                                    AS age_group
        FROM {TABLE}
        WHERE ALL_DIAGNOSES ILIKE '%urinary tract infection%'
          AND sex = 'F'
          AND age_in_years_float >= 15
    )
    SELECT
        age_group,
        COUNT(*)                                   AS total_uti_visits,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%pregnan%'
                 OR ALL_DIAGNOSES ILIKE '%obstetric%'
                 OR ALL_DIAGNOSES ILIKE '%antenatal%'  THEN 1 ELSE 0 END) AS with_pregnancy,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%diabetes%' THEN 1 ELSE 0 END) AS with_dm,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%chronic kidney%' THEN 1 ELSE 0 END) AS with_ckd,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%sepsis%'   THEN 1 ELSE 0 END) AS progressed_to_sepsis,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE ANY (
            '%pregnan%','%diabetes%','%chronic kidney%','%catheter%'
        ) THEN 1 ELSE 0 END)                       AS complicated_uti,
        ROUND(SUM(CASE WHEN ALL_DIAGNOSES ILIKE ANY (
            '%pregnan%','%diabetes%','%chronic kidney%','%catheter%'
        ) THEN 1.0 ELSE 0.0 END)
        / NULLIF(COUNT(*), 0) * 100, 1)            AS pct_complicated
    FROM female_uti
    GROUP BY age_group
    ORDER BY MIN(age_in_years_float)
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_uti_male_codx() -> pd.DataFrame:
    """Q3.5c Male UTI co-diagnoses."""
    return _run("q3_uti_male_codx", f"""
    WITH dx_long AS (
        SELECT pno, VALUE::STRING AS dx
        FROM {TABLE},
             LATERAL FLATTEN(INPUT => ARRAY_CONSTRUCT(
                 diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4,
                 diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
             )) f
        WHERE ALL_DIAGNOSES ILIKE '%urinary tract infection%'
          AND sex = 'M'
          AND VALUE NOT ILIKE '%urinary tract infection%'
          AND VALUE NOT ILIKE '%aortic aneurysm%'    -- exclude EMR default
          AND VALUE NOT ILIKE '%reduction defects%'  -- exclude EMR default
          AND VALUE IS NOT NULL
    )
    SELECT
        dx                                         AS co_diagnosis,
        COUNT(*)                                   AS occurrences
    FROM dx_long
    GROUP BY dx
    ORDER BY occurrences DESC
    LIMIT 15
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_toddler_uti() -> pd.DataFrame:
    """Q3.5d Toddler UTI by sex and year."""
    return _run("q3_toddler_uti", f"""
    SELECT
        sex,
        visit_year,
        COUNT(*)                                   AS visits,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%sepsis%'   THEN 1 ELSE 0 END) AS progressed_sepsis,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%kidney%'   THEN 1 ELSE 0 END) AS renal_involvement,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%fever%'    THEN 1 ELSE 0 END) AS with_fever,
        SUM(CASE WHEN is_severe_problem = TRUE THEN 1 ELSE 0 END)         AS severe_cases
    FROM {TABLE}
    WHERE ALL_DIAGNOSES ILIKE '%urinary tract infection%'
      AND age_in_years_float < 5
    GROUP BY sex, visit_year
    ORDER BY visit_year, sex
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_top_by_year() -> pd.DataFrame:
    """Q3.6a Top infectious diagnoses per year."""
    return _run("q3_top_by_year", f"""
    WITH dx_long AS (
        SELECT pno, visit_year, sex, age_in_years_float, VALUE::STRING AS dx
        FROM {TABLE},
             LATERAL FLATTEN(INPUT => ARRAY_CONSTRUCT(
                 diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4,
                 diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
             )) f
        WHERE (is_default_code_pair = FALSE OR is_default_code_pair IS NULL)
          AND VALUE IS NOT NULL
    ),
    infectious_dx AS (
        SELECT * FROM dx_long
        WHERE dx ILIKE ANY (
            '%pneumonia%','%sepsis%','%malaria%','%plasmodium%',
            '%urinary tract infection%','%gastritis%','%tonsil%',
            '%gastroenteritis%','%tuberculosis%','%gonococcal%',
            '%syphilis%','%chlamyd%','%typhoid%','%hepatitis%'
        )
        AND dx NOT ILIKE '%aortic%'
    )
    SELECT
        visit_year,
        dx                                         AS diagnosis,
        COUNT(*)                                   AS occurrences
    FROM infectious_dx
    GROUP BY visit_year, dx
    QUALIFY RANK() OVER (PARTITION BY visit_year ORDER BY COUNT(*) DESC) <= 10
    ORDER BY visit_year, occurrences DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_top_by_sex() -> pd.DataFrame:
    """Q3.6b Top infectious diagnoses per sex."""
    return _run("q3_top_by_sex", f"""
    WITH dx_long AS (
        SELECT pno, visit_year, sex, age_in_years_float, VALUE::STRING AS dx
        FROM {TABLE},
             LATERAL FLATTEN(INPUT => ARRAY_CONSTRUCT(
                 diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4,
                 diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
             )) f
        WHERE (is_default_code_pair = FALSE OR is_default_code_pair IS NULL)
          AND VALUE IS NOT NULL
    ),
    infectious_dx AS (
        SELECT * FROM dx_long
        WHERE dx ILIKE ANY (
            '%pneumonia%','%sepsis%','%malaria%','%plasmodium%',
            '%urinary tract infection%','%gastritis%','%tonsil%',
            '%gastroenteritis%','%tuberculosis%','%gonococcal%',
            '%syphilis%','%chlamyd%','%typhoid%','%hepatitis%'
        )
        AND dx NOT ILIKE '%aortic%'
    )
    SELECT
        sex,
        dx                                         AS diagnosis,
        COUNT(*)                                   AS occurrences
    FROM infectious_dx
    WHERE sex IN ('F','M')
    GROUP BY sex, dx
    QUALIFY RANK() OVER (PARTITION BY sex ORDER BY COUNT(*) DESC) <= 10
    ORDER BY sex, occurrences DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_top_by_age() -> pd.DataFrame:
    """Q3.6c Top infectious diagnoses per age group."""
    return _run("q3_top_by_age", f"""
    WITH dx_long AS (
        SELECT pno, visit_year, sex, age_in_years_float, VALUE::STRING AS dx
        FROM {TABLE},
             LATERAL FLATTEN(INPUT => ARRAY_CONSTRUCT(
                 diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4,
                 diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
             )) f
        WHERE (is_default_code_pair = FALSE OR is_default_code_pair IS NULL)
          AND VALUE IS NOT NULL
    ),
    infectious_dx AS (
        SELECT * FROM dx_long
        WHERE dx ILIKE ANY (
            '%pneumonia%','%sepsis%','%malaria%','%plasmodium%',
            '%urinary tract infection%','%gastritis%','%tonsil%',
            '%gastroenteritis%','%tuberculosis%','%gonococcal%',
            '%syphilis%','%chlamyd%','%typhoid%','%hepatitis%'
        )
        AND dx NOT ILIKE '%aortic%'
    )
    SELECT
        CASE
            WHEN age_in_years_float < 5  THEN 'Toddler (0-4)'
            WHEN age_in_years_float < 13 THEN 'Child (5-12)'
            WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
            WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
            WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
            WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
            WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
            ELSE '55+'
        END                                        AS age_group,
        dx                                         AS diagnosis,
        COUNT(*)                                   AS occurrences
    FROM infectious_dx
    WHERE age_in_years_float IS NOT NULL
    GROUP BY age_group, dx
    QUALIFY RANK() OVER (PARTITION BY age_group ORDER BY COUNT(*) DESC) <= 5
    ORDER BY MIN(age_in_years_float), occurrences DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_comorbidity() -> pd.DataFrame:
    """Q3.7 Infectious condition x comorbidity matrix."""
    return _run("q3_comorbidity", f"""
    WITH infectious_visits AS (
        SELECT
            pno, visit_year, age_in_years_float, sex, ALL_DIAGNOSES,
            CASE
                WHEN ALL_DIAGNOSES ILIKE '%pneumonia%'               THEN 'Pneumonia'
                WHEN ALL_DIAGNOSES ILIKE '%malaria%'
                  OR ALL_DIAGNOSES ILIKE '%plasmodium%'              THEN 'Malaria'
                WHEN ALL_DIAGNOSES ILIKE '%sepsis%'                  THEN 'Sepsis'
                WHEN ALL_DIAGNOSES ILIKE '%urinary tract infection%' THEN 'UTI'
                WHEN ALL_DIAGNOSES ILIKE '%gonococcal%'              THEN 'Gonococcal infection'
                WHEN ALL_DIAGNOSES ILIKE '%gastroenteritis%'         THEN 'Gastroenteritis'
                WHEN ALL_DIAGNOSES ILIKE '%tonsillitis%'             THEN 'Tonsillitis'
            END                                    AS infectious_condition
        FROM {TABLE}
        WHERE ALL_DIAGNOSES ILIKE ANY (
            '%pneumonia%','%malaria%','%plasmodium%','%sepsis%',
            '%urinary tract infection%','%gonococcal%',
            '%gastroenteritis%','%tonsillitis%'
        )
        AND (is_default_code_pair = FALSE OR is_default_code_pair IS NULL)
        AND ALL_DIAGNOSES IS NOT NULL
    )
    SELECT
        infectious_condition,
        COUNT(*)                                                      AS total_visits,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%anaemia%'       THEN 1 ELSE 0 END) AS with_anaemia,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%hypertension%'  THEN 1 ELSE 0 END) AS with_htn,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%diabetes%'      THEN 1 ELSE 0 END) AS with_dm,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%chronic kidney%' THEN 1 ELSE 0 END) AS with_ckd,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%heart failure%' THEN 1 ELSE 0 END) AS with_heart_failure,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%hiv%'           THEN 1 ELSE 0 END) AS with_hiv,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%sickle%'        THEN 1 ELSE 0 END) AS with_sickle_cell,
        -- Toddler-specific anaemia within each infectious disease
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%anaemia%'
                 AND age_in_years_float < 5 THEN 1 ELSE 0 END)       AS anaemia_in_toddlers,
        ROUND(SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%anaemia%'
                       THEN 1.0 ELSE 0.0 END)
              / NULLIF(COUNT(*), 0) * 100, 1)                         AS pct_with_anaemia,
        ROUND(SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%heart failure%'
                       THEN 1.0 ELSE 0.0 END)
              / NULLIF(COUNT(*), 0) * 100, 2)                         AS pct_with_heart_failure
    FROM infectious_visits
    WHERE infectious_condition IS NOT NULL
    GROUP BY infectious_condition
    ORDER BY total_visits DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_toddler_anaemia() -> pd.DataFrame:
    """Q3.7b Toddler anaemia: isolated vs comorbid, with sickle-cell coding."""
    return _run("q3_toddler_anaemia", f"""
    SELECT
        visit_year,
        COUNT(*)                                   AS toddler_anaemia_visits,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%malaria%'
                 OR ALL_DIAGNOSES ILIKE '%plasmodium%' THEN 1 ELSE 0 END) AS with_malaria,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%sepsis%'   THEN 1 ELSE 0 END) AS with_sepsis,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%pneumonia%' THEN 1 ELSE 0 END) AS with_pneumonia,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%sickle%'   THEN 1 ELSE 0 END) AS with_sickle_cell,
        SUM(CASE WHEN ALL_DIAGNOSES NOT ILIKE '%malaria%'
                 AND ALL_DIAGNOSES NOT ILIKE '%plasmodium%'
                 AND ALL_DIAGNOSES NOT ILIKE '%sepsis%'
                 AND ALL_DIAGNOSES NOT ILIKE '%pneumonia%'
                 THEN 1 ELSE 0 END)                AS isolated_anaemia,
        ROUND(SUM(CASE WHEN ALL_DIAGNOSES NOT ILIKE '%malaria%'
                 AND ALL_DIAGNOSES NOT ILIKE '%plasmodium%'
                 AND ALL_DIAGNOSES NOT ILIKE '%sepsis%'
                 AND ALL_DIAGNOSES NOT ILIKE '%pneumonia%'
                 THEN 1.0 ELSE 0.0 END)
              / NULLIF(COUNT(*), 0) * 100, 1)      AS pct_isolated
    FROM {TABLE}
    WHERE ALL_DIAGNOSES ILIKE '%anaemia%'
      AND age_in_years_float < 5
    GROUP BY visit_year
    ORDER BY visit_year
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_gonococcal() -> pd.DataFrame:
    """Q3.8 Gonococcal infection by year, sex and age, with pregnancy, PID, sepsis, HIV and origin."""
    return _run("q3_gonococcal", f"""
    SELECT
        visit_year,
        sex,
        CASE
            WHEN age_in_years_float < 18 THEN 'Adolescent (<18)'
            WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
            WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
            WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
            WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
            ELSE '55+'
        END                                        AS age_group,
        COUNT(*)                                   AS visits,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%pregnan%'
                 OR ALL_DIAGNOSES ILIKE '%antenatal%' THEN 1 ELSE 0 END) AS with_pregnancy,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%pelvic%'  THEN 1 ELSE 0 END) AS with_pid,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%sepsis%'  THEN 1 ELSE 0 END) AS with_sepsis,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%hiv%'     THEN 1 ELSE 0 END) AS with_hiv,
        SUM(CASE WHEN county = 'Nandi'              THEN 1 ELSE 0 END)   AS from_nandi,
        SUM(CASE WHEN county != 'Nandi'
                 AND county IS NOT NULL             THEN 1 ELSE 0 END)   AS from_outside_nandi,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY visit_year), 1)
                                                   AS pct_of_year
    FROM {TABLE}
    WHERE ALL_DIAGNOSES ILIKE '%gonococcal%'
      AND age_in_years_float IS NOT NULL
      AND sex IN ('F','M')
    GROUP BY visit_year, sex, age_group
    ORDER BY visit_year, visits DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_gonococcal_pregnancy() -> pd.DataFrame:
    """Q3.8b Gonococcal infection with pregnancy co-coded."""
    return _run("q3_gonococcal_pregnancy", f"""
    SELECT
        visit_year,
        CASE
            WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
            WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
            ELSE 'Adult (35+)'
        END                                        AS age_group,
        COUNT(*)                                   AS visits,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%sepsis%'  THEN 1 ELSE 0 END) AS with_sepsis,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%preterm%' THEN 1 ELSE 0 END) AS with_preterm,
        SUM(CASE WHEN is_severe_problem = TRUE THEN 1 ELSE 0 END)        AS severe_cases
    FROM {TABLE}
    WHERE ALL_DIAGNOSES ILIKE '%gonococcal%'
      AND ALL_DIAGNOSES ILIKE ANY ('%pregnan%','%antenatal%','%obstetric%')
    GROUP BY visit_year, age_group
    ORDER BY visit_year
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_ward_top5() -> pd.DataFrame:
    """N3.L Top 5 wards per disease (Nandi residents, 2024-2025)."""
    return _run("q3_ward_top5", f"""
    WITH d AS (
        SELECT ward, sub_county, unique_patient_id, ALL_DIAGNOSES, age_in_years_float
        FROM {TABLE}
        WHERE county = 'Nandi' AND ward IS NOT NULL AND visit_year IN (2024, 2025)
          AND ALL_DIAGNOSES IS NOT NULL
    ),
    f AS (
        SELECT 'Malaria' AS disease, * FROM d WHERE ALL_DIAGNOSES ILIKE ANY ('%malaria%', '%plasmodium%')
        UNION ALL SELECT 'Pneumonia',     * FROM d WHERE ALL_DIAGNOSES ILIKE '%pneumonia%'
        UNION ALL SELECT 'Sepsis',        * FROM d WHERE ALL_DIAGNOSES ILIKE '%sepsis%'
        UNION ALL SELECT 'Toddler sepsis',* FROM d WHERE ALL_DIAGNOSES ILIKE '%sepsis%' AND age_in_years_float < 5
        UNION ALL SELECT 'UTI',           * FROM d WHERE ALL_DIAGNOSES ILIKE '%urinary tract infection%'
        UNION ALL SELECT 'Gonococcal',    * FROM d WHERE ALL_DIAGNOSES ILIKE '%gonococcal%'
    )
    SELECT
        disease, ward, sub_county,
        COUNT(*)                                        AS visits,
        COUNT(DISTINCT unique_patient_id)               AS patients,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY disease), 1) AS pct_of_disease_visits
    FROM f
    GROUP BY disease, ward, sub_county
    QUALIFY ROW_NUMBER() OVER (PARTITION BY disease ORDER BY COUNT(*) DESC) <= 5
    ORDER BY disease, visits DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_ward_infectious() -> pd.DataFrame:
    """N3.L2 Top 5 wards by infectious visits, split by disease."""
    return _run("q3_ward_infectious", f"""
    WITH d AS (
        SELECT ward, sub_county, ALL_DIAGNOSES
        FROM {TABLE}
        WHERE county = 'Nandi' AND ward IS NOT NULL AND visit_year IN (2024, 2025)
          AND ALL_DIAGNOSES ILIKE ANY ('%pneumonia%','%sepsis%','%malaria%','%plasmodium%',
            '%urinary tract infection%','%gastritis%','%tonsil%',
            '%gastroenteritis%','%tuberculosis%','%gonococcal%',
            '%syphilis%','%chlamyd%','%typhoid%','%hepatitis%')
    ),
    top5 AS (
        SELECT ward FROM d GROUP BY ward ORDER BY COUNT(*) DESC LIMIT 5
    )
    SELECT
        d.ward, d.sub_county,
        COUNT(*)                                                                     AS infectious_visits,
        SUM(IFF(ALL_DIAGNOSES ILIKE ANY ('%malaria%','%plasmodium%'), 1, 0))          AS malaria,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%pneumonia%', 1, 0))                             AS pneumonia,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%sepsis%', 1, 0))                                AS sepsis,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%urinary tract infection%', 1, 0))               AS uti,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%gonococcal%', 1, 0))                            AS gonococcal
    FROM d JOIN top5 USING (ward)
    GROUP BY d.ward, d.sub_county
    ORDER BY infectious_visits DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_toddler_sepsis_age() -> pd.DataFrame:
    """N3.T1 Toddler sepsis by year of age, with infants split at 1 month."""
    return _run("q3_toddler_sepsis_age", f"""
    SELECT
        CASE
            WHEN age_in_years_float * 12 < 1  THEN '< 1 month'
            WHEN age_in_years_float < 1       THEN '1-11 months'
            ELSE FLOOR(age_in_years_float)::STRING || ' yr'
        END                                                    AS age_band,
        visit_year,
        CASE
            WHEN ALL_DIAGNOSES ILIKE '%without septic%'            THEN 'Sepsis without septic shock'
            WHEN ALL_DIAGNOSES ILIKE '%septic shock%'              THEN 'Septic shock'
            WHEN ALL_DIAGNOSES ILIKE '%severe sepsis%'             THEN 'Severe sepsis'
            WHEN ALL_DIAGNOSES ILIKE '%puerperal sepsis%'          THEN 'Puerperal sepsis'
            WHEN ALL_DIAGNOSES ILIKE '%neonatal sepsis%'           THEN 'Neonatal sepsis'
            WHEN ALL_DIAGNOSES ILIKE '%mild sepsis%'               THEN 'Mild sepsis'
            ELSE 'Other sepsis'
        END                                          AS sepsis_type,
        COUNT(*)                                               AS visits,
        SUM(IFF(sex = 'F', 1, 0))                              AS female,
        SUM(IFF(sex = 'M', 1, 0))                              AS male
    FROM {TABLE}
    WHERE ALL_DIAGNOSES ILIKE '%sepsis%'
      AND age_in_years_float IS NOT NULL AND age_in_years_float < 5
    GROUP BY age_band, visit_year, sepsis_type
    ORDER BY MIN(age_in_years_float), visit_year, visits DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_toddler_diarrhoea() -> pd.DataFrame:
    """N3.C Toddler diarrhoeal disease by year."""
    return _run("q3_toddler_diarrhoea", f"""
    SELECT
        visit_year,
        COUNT(*)                                                        AS toddler_visits,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%rotavirus%', 1, 0))               AS rotavirus,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%campylobacter%', 1, 0))           AS campylobacter,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%gastroenteritis%', 1, 0))         AS gastroenteritis_all,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%gastroenteritis%'
                AND ALL_DIAGNOSES NOT ILIKE '%rotavirus%'
                AND ALL_DIAGNOSES NOT ILIKE '%campylobacter%', 1, 0))   AS gastroenteritis_other
    FROM {TABLE}
    WHERE age_in_years_float < 5
      AND ALL_DIAGNOSES ILIKE ANY ('%gastroenteritis%', '%rotavirus%', '%campylobacter%')
    GROUP BY visit_year
    ORDER BY visit_year
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_malaria_pregnancy() -> pd.DataFrame:
    """N3.M1 Malaria in females 15-49 by age group and year, with pregnancy coded."""
    return _run("q3_malaria_pregnancy", f"""
    SELECT
        CASE
            WHEN age_in_years_float < 5  THEN 'Toddler (0-4)'
            WHEN age_in_years_float < 13 THEN 'Child (5-12)'
            WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
            WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
            WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
            WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
            WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
            ELSE '55+'
        END                                                             AS age_group,
        visit_year,
        COUNT(*)                                                           AS female_malaria_visits,
        SUM(IFF(ALL_DIAGNOSES ILIKE ANY ('%pregnan%','%antenatal%','%obstetric%'), 1, 0)) AS with_pregnancy,
        SUM(IFF(ALL_DIAGNOSES ILIKE ANY ('%pregnan%','%antenatal%','%obstetric%')
                AND ALL_DIAGNOSES ILIKE '%falciparum%', 1, 0))             AS pregnant_falciparum,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%anaemia%', 1, 0))                    AS with_anaemia,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%sepsis%', 1, 0))                     AS with_sepsis,
        SUM(IFF(is_severe_problem = TRUE, 1, 0))                           AS severe_cases
    FROM {TABLE}
    WHERE ALL_DIAGNOSES ILIKE ANY ('%malaria%', '%plasmodium%')
      AND sex = 'F' AND age_in_years_float >= 15 AND age_in_years_float < 50
    GROUP BY age_group, visit_year
    ORDER BY visit_year, MIN(age_in_years_float)
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_malaria_anaemia() -> pd.DataFrame:
    """N3.M2 Malaria visits that carry anaemia, by age."""
    return _run("q3_malaria_anaemia", f"""
    SELECT
        CASE
            WHEN age_in_years_float < 5  THEN 'Toddler (0-4)'
            WHEN age_in_years_float < 13 THEN 'Child (5-12)'
            WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
            WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
            WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
            WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
            WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
            ELSE '55+'
        END                                                             AS age_group,
        COUNT(*)                                                           AS malaria_visits,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%anaemia%', 1, 0))                    AS malaria_with_anaemia,
        ROUND(SUM(IFF(ALL_DIAGNOSES ILIKE '%anaemia%', 1, 0)) * 100.0 / NULLIF(COUNT(*), 0), 1) AS pct_malaria_with_anaemia,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%falciparum%', 1, 0))                 AS falciparum_visits,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%falciparum%' AND ALL_DIAGNOSES ILIKE '%anaemia%', 1, 0)) AS falciparum_with_anaemia
    FROM {TABLE}
    WHERE ALL_DIAGNOSES ILIKE ANY ('%malaria%', '%plasmodium%') AND age_in_years_float IS NOT NULL
    GROUP BY age_group
    ORDER BY MIN(age_in_years_float)
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_anaemia_malaria() -> pd.DataFrame:
    """N3.M2b Anaemia visits that carry malaria, by age."""
    return _run("q3_anaemia_malaria", f"""
    SELECT
        CASE
            WHEN age_in_years_float < 5  THEN 'Toddler (0-4)'
            WHEN age_in_years_float < 13 THEN 'Child (5-12)'
            WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
            WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
            WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
            WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
            WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
            ELSE '55+'
        END                                                             AS age_group,
        COUNT(*)                                                           AS anaemia_visits,
        SUM(IFF(ALL_DIAGNOSES ILIKE ANY ('%malaria%', '%plasmodium%'), 1, 0)) AS anaemia_with_malaria,
        ROUND(SUM(IFF(ALL_DIAGNOSES ILIKE ANY ('%malaria%', '%plasmodium%'), 1, 0)) * 100.0
              / NULLIF(COUNT(*), 0), 1)                                    AS pct_anaemia_with_malaria
    FROM {TABLE}
    WHERE ALL_DIAGNOSES ILIKE '%anaemia%' AND age_in_years_float IS NOT NULL
    GROUP BY age_group
    ORDER BY MIN(age_in_years_float)
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_artemisinin() -> pd.DataFrame:
    """N3.M3 Artemisinin-resistant cases by year, age and sex."""
    return _run("q3_artemisinin", f"""
    WITH dx_long AS (
        SELECT pno, visit_year, sex, age_in_years_float, VALUE::STRING AS dx
        FROM {TABLE},
             LATERAL FLATTEN(INPUT => ARRAY_CONSTRUCT(
                 diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4,
                 diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
             )) f
        WHERE VALUE::STRING ILIKE '%artemisinin%'
    )
    SELECT
        visit_year,
        CASE
            WHEN age_in_years_float < 5  THEN 'Toddler (0-4)'
            WHEN age_in_years_float < 13 THEN 'Child (5-12)'
            WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
            WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
            WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
            WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
            WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
            ELSE '55+'
        END                                                             AS age_group,
        sex,
        COUNT(*)                                                           AS dx_occurrences,
        COUNT(DISTINCT pno)                                                AS distinct_visits,
        ROUND(AVG(age_in_years_float), 1)                                  AS mean_age
    FROM dx_long
    WHERE age_in_years_float IS NOT NULL
    GROUP BY visit_year, age_group, sex
    ORDER BY visit_year, MIN(age_in_years_float)
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_mild_sepsis_codx() -> pd.DataFrame:
    """N3.S1 Top co-diagnoses on mild sepsis visits."""
    return _run("q3_mild_sepsis_codx", f"""
    WITH dx_long AS (
        SELECT pno, age_in_years_float, VALUE::STRING AS dx
        FROM {TABLE},
             LATERAL FLATTEN(INPUT => ARRAY_CONSTRUCT(
                 diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4,
                 diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
             )) f
        WHERE ALL_DIAGNOSES ILIKE '%mild sepsis%'
    )
    SELECT
        dx                                                       AS co_diagnosis,
        COUNT(DISTINCT pno)                                      AS visits,
        COUNT(DISTINCT IFF(age_in_years_float < 5, pno, NULL))   AS toddler_visits,
        ROUND(COUNT(DISTINCT pno) * 100.0
              / SUM(COUNT(DISTINCT pno)) OVER (), 1)             AS pct_of_codiagnoses
    FROM dx_long
    WHERE dx IS NOT NULL AND dx NOT ILIKE '%sepsis%'
    GROUP BY dx
    ORDER BY visits DESC
    LIMIT 15
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q3_uti_codx() -> pd.DataFrame:
    """N3.U1 UTI against other conditions, by sex and age."""
    return _run("q3_uti_codx", f"""
    SELECT
        sex,
        CASE
            WHEN age_in_years_float < 5  THEN 'Toddler (0-4)'
            WHEN age_in_years_float < 13 THEN 'Child (5-12)'
            WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
            WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
            WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
            WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
            WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
            ELSE '55+'
        END                                                             AS age_group,
        COUNT(*)                                                           AS uti_visits,
        SUM(IFF(ALL_DIAGNOSES ILIKE ANY ('%pregnan%','%antenatal%'), 1, 0)) AS with_pregnancy,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%diabetes%', 1, 0))                   AS with_diabetes,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%chronic kidney%', 1, 0))             AS with_ckd,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%heart failure%', 1, 0))              AS with_heart_failure,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%anaemia%', 1, 0))                    AS with_anaemia,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%hiv%', 1, 0))                        AS with_hiv,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%gonococcal%', 1, 0))                 AS with_gonococcal,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%hyperplasia of prostate%', 1, 0))    AS with_prostate,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%catheter%', 1, 0))                   AS with_catheter,
        SUM(IFF(ALL_DIAGNOSES ILIKE '%sepsis%', 1, 0))                     AS with_sepsis
    FROM {TABLE}
    WHERE ALL_DIAGNOSES ILIKE '%urinary tract infection%'
      AND sex IN ('F', 'M') AND age_in_years_float IS NOT NULL
    GROUP BY sex, age_group
    ORDER BY sex, MIN(age_in_years_float)
    """)

# =========================================================================== TAB 4 · NCDs
# Q4.x come from the NCD notes (ncd_queries_v3.sql); N4.x are new for the dashboard. All use the shared filter
# is_default_code_pair IS DISTINCT FROM TRUE, unpivot diagnosis_1..8 and label each diagnosis with one CASE.
# Conditions: hypertension, diabetes, chronic kidney disease, epilepsy, heart failure, stroke, asthma, COPD, cancer, sickle cell disease.
# Comparable full years are 2024 and 2025; 2023 and 2026 are shown as partial years and never compared.
# NEW queries (not in the earlier notes): N4.M1 q4_ward_patients, N4.M2 q4_ward_recurrent, N4.R1 q4_condition_return.

@st.cache_data(ttl=3600, show_spinner=False)
def q4_prevalence() -> pd.DataFrame:
    """Q4.1 NCD patients per condition, age group at first diagnosis, sex and origin (all years)."""
    return _run("q4_prevalence", f"""
    WITH labelled AS (
        SELECT
            unique_patient_id,
            visit_year,
            age_in_years_float,
            sex,
            county,
            ward,
            CASE
                WHEN diagnosis ILIKE '%hypertens%'
                    AND NOT (
                        diagnosis ILIKE ANY (
                            '%ocular%',
                            '%pulmonary%',
                            '%portal%',
                            '%intracranial%',
                            '%eclampsia%',
                            '%gestational%',
                            '%pregnan%',
                            '%puerper%',
                            '%pih%'
                        )
                    )
                THEN 'Hypertension'
                WHEN diagnosis ILIKE '%diabetes%'
                    AND NOT (
                        diagnosis ILIKE ANY ('%insipidus%', '%gestational%', '%in pregnancy%', '%without associated diabetes%')
                    )
                THEN 'Diabetes'
                WHEN diagnosis ILIKE ANY ('%chronic kidney%', '%chronic renal%', '%end-stage renal%', '%end stage renal%') THEN 'Chronic kidney disease'
                WHEN diagnosis ILIKE '%epilep%' THEN 'Epilepsy'
                WHEN diagnosis ILIKE ANY ('%heart failure%', '%cardiac failure%') THEN 'Heart failure'
                WHEN diagnosis ILIKE ANY ('%stroke%', '%cerebrovascular%') THEN 'Stroke / cerebrovascular disease'
                WHEN diagnosis ILIKE '%asthma%' THEN 'Asthma'
                WHEN diagnosis ILIKE ANY ('%chronic obstructive%', '%copd%') THEN 'COPD'
                WHEN diagnosis ILIKE ANY ('%malignant%', '%carcinoma%', '%cancer%', '%lymphoma%', '%leukaemia%', '%leukemia%', '%sarcoma%', '%myeloma%')
                    AND NOT (
                        diagnosis ILIKE ANY ('%benign%', '%in situ%')
                    )
                THEN 'Cancer'
                WHEN diagnosis ILIKE ANY ('%sickle cell%', '%sickle-cell%') AND diagnosis NOT ILIKE '%trait%' THEN 'Sickle cell disease'
            END AS condition_name
        FROM {TABLE}
        UNPIVOT (diagnosis FOR slot IN (diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4, diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8))
        WHERE
            is_default_code_pair IS DISTINCT FROM TRUE AND age_in_years_float IS NOT NULL
    ), matches AS (
        SELECT
            condition_name,
            unique_patient_id,
            visit_year,
            age_in_years_float,
            sex,
            county,
            ROW_NUMBER() OVER (PARTITION BY condition_name, unique_patient_id ORDER BY visit_year, age_in_years_float) AS rn
        FROM labelled
        WHERE
            condition_name IS NOT NULL
    ), first_occurrence AS (
        SELECT
            condition_name,
            unique_patient_id,
            visit_year AS first_diagnosis_year,
            sex,
            county,
            CASE
                WHEN age_in_years_float < 5 THEN 'Toddler (0-4)'
                WHEN age_in_years_float < 13 THEN 'Child (5-12)'
                WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
                WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
                WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
                WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
                WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
                ELSE '55+'
            END AS age_group_at_first_diagnosis
        FROM matches
        WHERE
            rn = 1
    )
    SELECT
        condition_name,
        age_group_at_first_diagnosis,
        sex,
        COUNT(DISTINCT unique_patient_id) AS patients,
        SUM(CASE WHEN county = 'Nandi' THEN 1 ELSE 0 END) AS from_nandi,
        SUM(CASE WHEN county <> 'Nandi' AND county IS NOT NULL THEN 1 ELSE 0 END) AS from_outside_nandi
    FROM first_occurrence
    GROUP BY
        condition_name,
        age_group_at_first_diagnosis,
        sex
    ORDER BY
        condition_name,
        MIN(age_group_at_first_diagnosis)
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_timeseries() -> pd.DataFrame:
    """Q4.1b Distinct patients per condition per year (all years; 2023 and 2026 are partial)."""
    return _run("q4_timeseries", f"""
    WITH labelled AS (
        SELECT
            unique_patient_id,
            visit_year,
            age_in_years_float,
            sex,
            county,
            ward,
            CASE
                WHEN diagnosis ILIKE '%hypertens%'
                    AND NOT (
                        diagnosis ILIKE ANY (
                            '%ocular%',
                            '%pulmonary%',
                            '%portal%',
                            '%intracranial%',
                            '%eclampsia%',
                            '%gestational%',
                            '%pregnan%',
                            '%puerper%',
                            '%pih%'
                        )
                    )
                THEN 'Hypertension'
                WHEN diagnosis ILIKE '%diabetes%'
                    AND NOT (
                        diagnosis ILIKE ANY ('%insipidus%', '%gestational%', '%in pregnancy%', '%without associated diabetes%')
                    )
                THEN 'Diabetes'
                WHEN diagnosis ILIKE ANY ('%chronic kidney%', '%chronic renal%', '%end-stage renal%', '%end stage renal%') THEN 'Chronic kidney disease'
                WHEN diagnosis ILIKE '%epilep%' THEN 'Epilepsy'
                WHEN diagnosis ILIKE ANY ('%heart failure%', '%cardiac failure%') THEN 'Heart failure'
                WHEN diagnosis ILIKE ANY ('%stroke%', '%cerebrovascular%') THEN 'Stroke / cerebrovascular disease'
                WHEN diagnosis ILIKE '%asthma%' THEN 'Asthma'
                WHEN diagnosis ILIKE ANY ('%chronic obstructive%', '%copd%') THEN 'COPD'
                WHEN diagnosis ILIKE ANY ('%malignant%', '%carcinoma%', '%cancer%', '%lymphoma%', '%leukaemia%', '%leukemia%', '%sarcoma%', '%myeloma%')
                    AND NOT (
                        diagnosis ILIKE ANY ('%benign%', '%in situ%')
                    )
                THEN 'Cancer'
                WHEN diagnosis ILIKE ANY ('%sickle cell%', '%sickle-cell%') AND diagnosis NOT ILIKE '%trait%' THEN 'Sickle cell disease'
            END AS condition_name
        FROM {TABLE}
        UNPIVOT (diagnosis FOR slot IN (diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4, diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8))
        WHERE
            is_default_code_pair IS DISTINCT FROM TRUE
    )
    SELECT
        condition_name,
        visit_year,
        COUNT(DISTINCT unique_patient_id) AS patients_seen
    FROM labelled
    WHERE
        condition_name IS NOT NULL
    GROUP BY
        condition_name,
        visit_year
    ORDER BY
        condition_name,
        visit_year
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_comorbidity() -> pd.DataFrame:
    """Q4.2 Comorbidity pairs by age group and sex (a floor: most visits record one diagnosis)."""
    return _run("q4_comorbidity", f"""
    WITH flags AS (
        SELECT
            unique_patient_id,
            MAX(sex) AS sex,
            MIN(age_in_years_float) AS age_at_first_visit,
            MAX(
                CASE
                    WHEN (
                            ALL_DIAGNOSES ILIKE '%hypertens%'
                            AND NOT (
                                ALL_DIAGNOSES ILIKE ANY (
                                    '%ocular hypertens%',
                                    '%pulmonary hypertens%',
                                    '%portal hypertens%',
                                    '%intracranial hypertens%',
                                    '%eclampsia%',
                                    '%gestational hypertension%',
                                    '%pregnancy-induced hypertension%',
                                    '%complicating pregnancy%',
                                    '%puerper%',
                                    '%pih%'
                                )
                            )
                        )
                    THEN 1
                    ELSE 0
                END
            ) AS has_hypertension,
            MAX(
                CASE
                    WHEN (
                            ALL_DIAGNOSES ILIKE '%diabetes%'
                            AND NOT (
                                ALL_DIAGNOSES ILIKE ANY ('%insipidus%', '%gestational diabetes%', '%in pregnancy%', '%without associated diabetes%')
                            )
                        )
                    THEN 1
                    ELSE 0
                END
            ) AS has_diabetes,
            MAX(
                CASE
                    WHEN ALL_DIAGNOSES ILIKE ANY ('%chronic kidney%', '%chronic renal%', '%end-stage renal%', '%end stage renal%') THEN 1
                    ELSE 0
                END
            ) AS has_ckd,
            MAX(CASE WHEN ALL_DIAGNOSES ILIKE '%epilep%' THEN 1 ELSE 0 END) AS has_epilepsy,
            MAX(CASE WHEN ALL_DIAGNOSES ILIKE ANY ('%heart failure%', '%cardiac failure%') THEN 1 ELSE 0 END) AS has_heart_failure
        FROM {TABLE}
        WHERE
            is_default_code_pair IS DISTINCT FROM TRUE AND ALL_DIAGNOSES IS NOT NULL
        GROUP BY
            unique_patient_id
    ), tagged AS (
        SELECT
            *,
            CASE
                WHEN age_at_first_visit < 5 THEN 'Toddler (0-4)'
                WHEN age_at_first_visit < 13 THEN 'Child (5-12)'
                WHEN age_at_first_visit < 18 THEN 'Adolescent (13-17)'
                WHEN age_at_first_visit < 25 THEN 'Youth (18-24)'
                WHEN age_at_first_visit < 35 THEN 'Young Adult (25-34)'
                WHEN age_at_first_visit < 45 THEN 'Adult (35-44)'
                WHEN age_at_first_visit < 55 THEN 'Middle Age (45-54)'
                ELSE '55+'
            END AS age_group
        FROM flags
        WHERE
            age_at_first_visit IS NOT NULL
    )
    SELECT
        age_group,
        sex,
        SUM(has_hypertension * has_diabetes) AS hypertension_and_diabetes,
        SUM(has_hypertension * has_ckd) AS hypertension_and_ckd,
        SUM(has_diabetes * has_ckd) AS diabetes_and_ckd,
        SUM(has_hypertension * has_epilepsy) AS hypertension_and_epilepsy,
        SUM(has_hypertension * has_heart_failure) AS hypertension_and_heart_failure,
        SUM(has_diabetes * has_heart_failure) AS diabetes_and_heart_failure
    FROM tagged
    GROUP BY
        age_group,
        sex
    ORDER BY
        MIN(age_at_first_visit),
        sex
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_returning() -> pd.DataFrame:
    """Q4.4 Returning NCD patients: age, sex, origin, visits, NCD share of visits, comorbidity."""
    return _run("q4_returning", f"""
    WITH ncd_visits AS (
        SELECT
            unique_patient_id,
            visit_year,
            age_in_years_float,
            sex,
            county,
            CASE
                WHEN (
                        ALL_DIAGNOSES ILIKE '%hypertens%'
                        AND NOT (
                            ALL_DIAGNOSES ILIKE ANY (
                                '%ocular hypertens%',
                                '%pulmonary hypertens%',
                                '%portal hypertens%',
                                '%intracranial hypertens%',
                                '%eclampsia%',
                                '%gestational hypertension%',
                                '%pregnancy-induced hypertension%',
                                '%complicating pregnancy%',
                                '%puerper%',
                                '%pih%'
                            )
                        )
                    )
                    OR (
                        ALL_DIAGNOSES ILIKE '%diabetes%'
                        AND NOT (
                            ALL_DIAGNOSES ILIKE ANY ('%insipidus%', '%gestational diabetes%', '%in pregnancy%', '%without associated diabetes%')
                        )
                    )
                    OR ALL_DIAGNOSES ILIKE ANY ('%chronic kidney%', '%chronic renal%', '%end-stage renal%', '%end stage renal%')
                    OR ALL_DIAGNOSES ILIKE '%epilep%'
                    OR ALL_DIAGNOSES ILIKE ANY ('%heart failure%', '%cardiac failure%')
                    OR ALL_DIAGNOSES ILIKE ANY ('%stroke%', '%cerebrovascular%')
                    OR ALL_DIAGNOSES ILIKE '%asthma%'
                    OR ALL_DIAGNOSES ILIKE ANY ('%chronic obstructive%', '%copd%')
                    OR (
                        ALL_DIAGNOSES ILIKE ANY ('%malignant%', '%carcinoma%', '%cancer%', '%lymphoma%', '%leukaemia%', '%leukemia%', '%sarcoma%', '%myeloma%')
                        AND NOT (
                            ALL_DIAGNOSES ILIKE ANY ('%benign%', '%in situ%')
                        )
                    )
                    OR (
                        ALL_DIAGNOSES ILIKE ANY ('%sickle cell%', '%sickle-cell%') AND ALL_DIAGNOSES NOT ILIKE '%trait%'
                    )
                THEN 1
                ELSE 0
            END AS is_ncd_visit
        FROM {TABLE}
        WHERE
            is_default_code_pair IS DISTINCT FROM TRUE AND ALL_DIAGNOSES IS NOT NULL
    ), patient_ncd_summary AS (
        SELECT
            unique_patient_id,
            COUNT(DISTINCT visit_year) AS years_seen,
            MIN(visit_year) AS first_year,
            MAX(visit_year) AS last_year,
            MIN(age_in_years_float) AS age_at_first_visit,
            MAX(sex) AS sex,
            MAX(county) AS county,
            COUNT(*) AS total_visits,
            SUM(is_ncd_visit) AS ncd_visits,
            COUNT(*) - SUM(is_ncd_visit) AS non_ncd_visits,
            ROUND(SUM(is_ncd_visit) * 100.0 / NULLIF(COUNT(*), 0), 1) AS pct_visits_ncd
        FROM ncd_visits
        GROUP BY
            unique_patient_id
        HAVING
            years_seen > 1 AND SUM(is_ncd_visit) > 0 AND NOT MIN(age_in_years_float) IS NULL
    ), comorbidity_count AS (
        SELECT
            unique_patient_id,
            MAX(
                CASE
                    WHEN (
                            ALL_DIAGNOSES ILIKE '%hypertens%'
                            AND NOT (
                                ALL_DIAGNOSES ILIKE ANY (
                                    '%ocular hypertens%',
                                    '%pulmonary hypertens%',
                                    '%portal hypertens%',
                                    '%intracranial hypertens%',
                                    '%eclampsia%',
                                    '%gestational hypertension%',
                                    '%pregnancy-induced hypertension%',
                                    '%complicating pregnancy%',
                                    '%puerper%',
                                    '%pih%'
                                )
                            )
                        )
                    THEN 1
                    ELSE 0
                END
            ) + MAX(
                CASE
                    WHEN (
                            ALL_DIAGNOSES ILIKE '%diabetes%'
                            AND NOT (
                                ALL_DIAGNOSES ILIKE ANY ('%insipidus%', '%gestational diabetes%', '%in pregnancy%', '%without associated diabetes%')
                            )
                        )
                    THEN 1
                    ELSE 0
                END
            ) + MAX(
                CASE
                    WHEN ALL_DIAGNOSES ILIKE ANY ('%chronic kidney%', '%chronic renal%', '%end-stage renal%', '%end stage renal%') THEN 1
                    ELSE 0
                END
            ) + MAX(CASE WHEN ALL_DIAGNOSES ILIKE '%epilep%' THEN 1 ELSE 0 END) + MAX(CASE WHEN ALL_DIAGNOSES ILIKE ANY ('%heart failure%', '%cardiac failure%') THEN 1 ELSE 0 END) + MAX(CASE WHEN ALL_DIAGNOSES ILIKE ANY ('%stroke%', '%cerebrovascular%') THEN 1 ELSE 0 END) + MAX(CASE WHEN ALL_DIAGNOSES ILIKE '%asthma%' THEN 1 ELSE 0 END) + MAX(CASE WHEN ALL_DIAGNOSES ILIKE ANY ('%chronic obstructive%', '%copd%') THEN 1 ELSE 0 END) + MAX(
                CASE
                    WHEN (
                            ALL_DIAGNOSES ILIKE ANY ('%malignant%', '%carcinoma%', '%cancer%', '%lymphoma%', '%leukaemia%', '%leukemia%', '%sarcoma%', '%myeloma%')
                            AND NOT (
                                ALL_DIAGNOSES ILIKE ANY ('%benign%', '%in situ%')
                            )
                        )
                    THEN 1
                    ELSE 0
                END
            ) + MAX(
                CASE
                    WHEN (
                            ALL_DIAGNOSES ILIKE ANY ('%sickle cell%', '%sickle-cell%') AND ALL_DIAGNOSES NOT ILIKE '%trait%'
                        )
                    THEN 1
                    ELSE 0
                END
            ) AS n_ncds_ever
        FROM {TABLE}
        WHERE
            is_default_code_pair IS DISTINCT FROM TRUE AND ALL_DIAGNOSES IS NOT NULL
        GROUP BY
            unique_patient_id
    )
    SELECT
        CASE
            WHEN p.age_at_first_visit < 5 THEN 'Toddler (0-4)'
            WHEN p.age_at_first_visit < 13 THEN 'Child (5-12)'
            WHEN p.age_at_first_visit < 18 THEN 'Adolescent (13-17)'
            WHEN p.age_at_first_visit < 25 THEN 'Youth (18-24)'
            WHEN p.age_at_first_visit < 35 THEN 'Young Adult (25-34)'
            WHEN p.age_at_first_visit < 45 THEN 'Adult (35-44)'
            WHEN p.age_at_first_visit < 55 THEN 'Middle Age (45-54)'
            ELSE '55+'
        END AS age_group,
        p.sex,
        CASE WHEN p.county = 'Nandi' THEN 'From Nandi' ELSE 'Outside Nandi' END AS origin,
        COUNT(*) AS patients,
        ROUND(AVG(p.total_visits), 1) AS avg_total_visits,
        ROUND(AVG(p.pct_visits_ncd), 1) AS avg_pct_visits_ncd,
        SUM(CASE WHEN c.n_ncds_ever > 1 THEN 1 ELSE 0 END) AS comorbid_patients
    FROM patient_ncd_summary AS p
    JOIN comorbidity_count AS c
        ON p.unique_patient_id = c.unique_patient_id
    GROUP BY
        age_group,
        p.sex,
        origin
    ORDER BY
        age_group,
        p.sex,
        origin
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_anaemia_subtypes() -> pd.DataFrame:
    """Q4.6 Anaemia subtypes and NCD co-occurrence (why anaemia is not counted as an NCD)."""
    return _run("q4_anaemia_subtypes", f"""
    SELECT
        CASE
            WHEN ALL_DIAGNOSES ILIKE '%iron deficiency%' THEN 'Iron deficiency anaemia (nutritional — not NCD per GBD)'
            WHEN ALL_DIAGNOSES ILIKE '%haemolytic%' OR ALL_DIAGNOSES ILIKE '%hemolytic%' THEN 'Haemolytic anaemia'
            WHEN ALL_DIAGNOSES ILIKE '%anaemia of%' THEN 'Anaemia of chronic disease'
            ELSE 'Anaemia, unspecified/other'
        END AS anaemia_subtype,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%chronic kidney%' THEN 1 ELSE 0 END) AS co_coded_with_ckd,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%diabetes%' THEN 1 ELSE 0 END) AS co_coded_with_diabetes,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%malaria%' OR ALL_DIAGNOSES ILIKE '%plasmodium%' THEN 1 ELSE 0 END) AS co_coded_with_malaria,
        SUM(CASE WHEN ALL_DIAGNOSES ILIKE '%sepsis%' THEN 1 ELSE 0 END) AS co_coded_with_sepsis,
        COUNT(*) AS total_visits
    FROM {TABLE}
    WHERE
        ALL_DIAGNOSES ILIKE '%anaemia%' AND is_default_code_pair IS DISTINCT FROM TRUE
    GROUP BY
        anaemia_subtype
    ORDER BY
        total_visits DESC
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_sickle_crisis() -> pd.DataFrame:
    """Q4.7 Sickle cell disease patients by age group, sex and crisis status (one row per patient)."""
    return _run("q4_sickle_crisis", f"""
    WITH pt AS (
        SELECT
            unique_patient_id,
            MAX(age_in_years_float) AS age,
            MODE(sex) AS sex,
            MAX(IFF(ALL_DIAGNOSES ILIKE '%sickle cell disease with crisis%', 1, 0)) AS had_crisis,
            MAX(IFF(ALL_DIAGNOSES ILIKE '%sickle cell disease without crisis%', 1, 0)) AS had_no_crisis
        FROM {TABLE}
        WHERE
            is_default_code_pair IS DISTINCT FROM TRUE
            AND ALL_DIAGNOSES ILIKE ANY ('%sickle cell%', '%sickle-cell%')
            AND ALL_DIAGNOSES NOT ILIKE '%trait%'
            AND age_in_years_float IS NOT NULL
        GROUP BY
            unique_patient_id
    )
    SELECT
        CASE
            WHEN age < 5 THEN 'Toddler (0-4)'
            WHEN age < 13 THEN 'Child (5-12)'
            WHEN age < 18 THEN 'Adolescent (13-17)'
            WHEN age < 25 THEN 'Youth (18-24)'
            WHEN age < 35 THEN 'Young Adult (25-34)'
            WHEN age < 45 THEN 'Adult (35-44)'
            WHEN age < 55 THEN 'Middle Age (45-54)'
            ELSE '55+'
        END AS age_group,
        sex,
        CASE
            WHEN had_crisis = 1 THEN 'With crisis'
            WHEN had_no_crisis = 1 THEN 'Without crisis'
            ELSE 'Sickle cell disorder (unspecified)'
        END AS crisis_status,
        COUNT(*) AS patients
    FROM pt
    GROUP BY
        age_group,
        sex,
        crisis_status
    ORDER BY
        MIN(age),
        sex
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_sickle_trait() -> pd.DataFrame:
    """Q4.7b Sickle cell trait carriers, reported separately."""
    return _run("q4_sickle_trait", f"""
    SELECT
        CASE WHEN age_in_years_float < 18 THEN 'Under 18' ELSE '18+' END AS age_group,
        sex,
        COUNT(DISTINCT unique_patient_id) AS carriers_identified
    FROM {TABLE}
    WHERE
        is_default_code_pair IS DISTINCT FROM TRUE
        AND ALL_DIAGNOSES ILIKE '%sickle cell trait%'
        AND age_in_years_float IS NOT NULL
    GROUP BY
        age_group,
        sex
    ORDER BY
        age_group,
        sex
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_sickle_returning() -> pd.DataFrame:
    """Q4.7c Returning sickle cell disease patients."""
    return _run("q4_sickle_returning", f"""
    WITH scd_patients AS (
        SELECT DISTINCT
            unique_patient_id
        FROM {TABLE}
        WHERE
            is_default_code_pair IS DISTINCT FROM TRUE
            AND ALL_DIAGNOSES IS NOT NULL
            AND ALL_DIAGNOSES ILIKE '%sickle cell disease%'
    ), visit_counts AS (
        SELECT
            unique_patient_id,
            COUNT(DISTINCT visit_year) AS years_seen,
            COUNT(*) AS total_visits,
            MAX(county) AS county
        FROM {TABLE}
        WHERE
            is_default_code_pair IS DISTINCT FROM TRUE AND ALL_DIAGNOSES IS NOT NULL
        GROUP BY
            unique_patient_id
    )
    SELECT
        CASE WHEN vc.county = 'Nandi' THEN 'From Nandi' ELSE 'Outside Nandi' END AS origin,
        COUNT(*) AS patients,
        ROUND(AVG(vc.years_seen), 2) AS avg_years_seen,
        ROUND(AVG(vc.total_visits), 2) AS avg_total_visits,
        SUM(CASE WHEN vc.years_seen > 1 THEN 1 ELSE 0 END) AS multi_year_patients
    FROM scd_patients AS sp
    JOIN visit_counts AS vc
        ON sp.unique_patient_id = vc.unique_patient_id
    GROUP BY
        origin
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_baseline() -> pd.DataFrame:
    """Q4.8 Baseline OPD attendance by age group and sex."""
    return _run("q4_baseline", f"""
    SELECT
        CASE
            WHEN age_in_years_float < 5 THEN 'Toddler (0-4)'
            WHEN age_in_years_float < 13 THEN 'Child (5-12)'
            WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
            WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
            WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
            WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
            WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
            ELSE '55+'
        END AS age_group,
        sex,
        COUNT(DISTINCT unique_patient_id) AS distinct_patients,
        COUNT(*) AS total_visits
    FROM {TABLE}
    WHERE
        is_default_code_pair IS DISTINCT FROM TRUE
        AND ALL_DIAGNOSES IS NOT NULL
        AND age_in_years_float IS NOT NULL
        AND sex IN ('F', 'M')
    GROUP BY
        age_group,
        sex
    ORDER BY
        MIN(age_in_years_float),
        sex
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_female_gap() -> pd.DataFrame:
    """Q4.8b Female share of each condition vs the OPD baseline, by age group."""
    return _run("q4_female_gap", f"""
    WITH baseline AS (
        SELECT
            CASE
                WHEN age_in_years_float < 5 THEN 'Toddler (0-4)'
                WHEN age_in_years_float < 13 THEN 'Child (5-12)'
                WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
                WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
                WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
                WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
                WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
                ELSE '55+'
            END AS age_group,
            COUNT(DISTINCT CASE WHEN sex = 'F' THEN unique_patient_id END) AS female_patients,
            COUNT(DISTINCT unique_patient_id) AS total_patients
        FROM {TABLE}
        WHERE
            is_default_code_pair IS DISTINCT FROM TRUE
            AND ALL_DIAGNOSES IS NOT NULL
            AND age_in_years_float IS NOT NULL
            AND sex IN ('F', 'M')
        GROUP BY
            age_group
    ), labelled AS (
        SELECT
            unique_patient_id,
            visit_year,
            age_in_years_float,
            sex,
            county,
            ward,
            CASE
                WHEN diagnosis ILIKE '%hypertens%'
                    AND NOT (
                        diagnosis ILIKE ANY (
                            '%ocular%',
                            '%pulmonary%',
                            '%portal%',
                            '%intracranial%',
                            '%eclampsia%',
                            '%gestational%',
                            '%pregnan%',
                            '%puerper%',
                            '%pih%'
                        )
                    )
                THEN 'Hypertension'
                WHEN diagnosis ILIKE '%diabetes%'
                    AND NOT (
                        diagnosis ILIKE ANY ('%insipidus%', '%gestational%', '%in pregnancy%', '%without associated diabetes%')
                    )
                THEN 'Diabetes'
                WHEN diagnosis ILIKE ANY ('%chronic kidney%', '%chronic renal%', '%end-stage renal%', '%end stage renal%') THEN 'Chronic kidney disease'
                WHEN diagnosis ILIKE '%epilep%' THEN 'Epilepsy'
                WHEN diagnosis ILIKE ANY ('%heart failure%', '%cardiac failure%') THEN 'Heart failure'
                WHEN diagnosis ILIKE ANY ('%stroke%', '%cerebrovascular%') THEN 'Stroke / cerebrovascular disease'
                WHEN diagnosis ILIKE '%asthma%' THEN 'Asthma'
                WHEN diagnosis ILIKE ANY ('%chronic obstructive%', '%copd%') THEN 'COPD'
                WHEN diagnosis ILIKE ANY ('%malignant%', '%carcinoma%', '%cancer%', '%lymphoma%', '%leukaemia%', '%leukemia%', '%sarcoma%', '%myeloma%')
                    AND NOT (
                        diagnosis ILIKE ANY ('%benign%', '%in situ%')
                    )
                THEN 'Cancer'
                WHEN diagnosis ILIKE ANY ('%sickle cell%', '%sickle-cell%') AND diagnosis NOT ILIKE '%trait%' THEN 'Sickle cell disease'
            END AS condition_name
        FROM {TABLE}
        UNPIVOT (diagnosis FOR slot IN (diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4, diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8))
        WHERE
            is_default_code_pair IS DISTINCT FROM TRUE AND age_in_years_float IS NOT NULL AND sex IN ('F', 'M')
    ), ncd_patients AS (
        SELECT DISTINCT
            condition_name,
            unique_patient_id,
            sex,
            CASE
                WHEN age_in_years_float < 5 THEN 'Toddler (0-4)'
                WHEN age_in_years_float < 13 THEN 'Child (5-12)'
                WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
                WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
                WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
                WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
                WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
                ELSE '55+'
            END AS age_group
        FROM labelled
        WHERE
            condition_name IS NOT NULL
    ), ncd_summary AS (
        SELECT
            condition_name,
            age_group,
            COUNT(DISTINCT CASE WHEN sex = 'F' THEN unique_patient_id END) AS female_ncd_patients,
            COUNT(DISTINCT unique_patient_id) AS total_ncd_patients
        FROM ncd_patients
        GROUP BY
            condition_name,
            age_group
    )
    SELECT
        n.condition_name,
        n.age_group,
        ROUND(n.female_ncd_patients * 100.0 / NULLIF(n.total_ncd_patients, 0), 1) AS pct_female_this_condition,
        ROUND(b.female_patients * 100.0 / NULLIF(b.total_patients, 0), 1) AS pct_female_baseline_opd,
        ROUND(
            (
                n.female_ncd_patients * 100.0 / NULLIF(n.total_ncd_patients, 0)
            ) - (
                b.female_patients * 100.0 / NULLIF(b.total_patients, 0)
            ),
            1
        ) AS gap_vs_baseline_pts
    FROM ncd_summary AS n
    JOIN baseline AS b
        ON n.age_group = b.age_group
    ORDER BY
        n.condition_name,
        CASE n.age_group
            WHEN 'Toddler (0-4)' THEN 1
            WHEN 'Child (5-12)' THEN 2
            WHEN 'Adolescent (13-17)' THEN 3
            WHEN 'Youth (18-24)' THEN 4
            WHEN 'Young Adult (25-34)' THEN 5
            WHEN 'Adult (35-44)' THEN 6
            WHEN 'Middle Age (45-54)' THEN 7
            ELSE 8
        END
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_htn_pregnancy() -> pd.DataFrame:
    """Q4.9 Women's hypertension visits and the pregnancy-related share."""
    return _run("q4_htn_pregnancy", f"""
    SELECT
        CASE
            WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
            WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
            WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
            WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
            ELSE '55+'
        END AS age_group,
        COUNT(*) AS total_hypertension_visits_f,
        SUM(
            CASE
                WHEN ALL_DIAGNOSES ILIKE '%pre-eclampsia%'
                    OR ALL_DIAGNOSES ILIKE '%preeclampsia%'
                    OR ALL_DIAGNOSES ILIKE '%eclampsia%'
                    OR ALL_DIAGNOSES ILIKE '%gestational hypertension%'
                    OR ALL_DIAGNOSES ILIKE '%pregnancy-induced hypertension%'
                    OR ALL_DIAGNOSES ILIKE '%complicating pregnancy%'
                    OR ALL_DIAGNOSES ILIKE '%pih%'
                THEN 1
                ELSE 0
            END
        ) AS pregnancy_related_htn,
        SUM(
            CASE
                WHEN ALL_DIAGNOSES ILIKE '%pregnan%' OR ALL_DIAGNOSES ILIKE '%antenatal%' OR ALL_DIAGNOSES ILIKE '%obstetric%' THEN 1
                ELSE 0
            END
        ) AS any_pregnancy_context,
        ROUND(
            SUM(
                CASE
                    WHEN ALL_DIAGNOSES ILIKE '%pregnan%' OR ALL_DIAGNOSES ILIKE '%antenatal%' OR ALL_DIAGNOSES ILIKE '%obstetric%' THEN 1.0
                    ELSE 0.0
                END
            ) / NULLIF(COUNT(*), 0) * 100,
            1
        ) AS pct_with_any_pregnancy_context
    FROM {TABLE}
    WHERE
        ALL_DIAGNOSES ILIKE '%hypertens%'
        AND sex = 'F'
        AND age_in_years_float >= 18
        AND is_default_code_pair IS DISTINCT FROM TRUE
    GROUP BY
        age_group
    ORDER BY
        MIN(age_in_years_float)
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_htn_chronic() -> pd.DataFrame:
    """Q4.10 Chronic hypertension only (pregnancy-related cases removed), by age group and sex."""
    return _run("q4_htn_chronic", f"""
    SELECT
        CASE
            WHEN age_in_years_float < 5 THEN 'Toddler (0-4)'
            WHEN age_in_years_float < 13 THEN 'Child (5-12)'
            WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
            WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
            WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
            WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
            WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
            ELSE '55+'
        END AS age_group,
        sex,
        COUNT(DISTINCT unique_patient_id) AS patients_chronic_hypertension_only
    FROM {TABLE}
    WHERE
        ALL_DIAGNOSES ILIKE '%hypertens%'
        AND ALL_DIAGNOSES NOT ILIKE '%ocular hypertens%'
        AND ALL_DIAGNOSES NOT ILIKE '%pulmonary hypertens%'
        AND ALL_DIAGNOSES NOT ILIKE '%portal hypertens%'
        AND ALL_DIAGNOSES NOT ILIKE '%intracranial hypertens%'
        AND ALL_DIAGNOSES NOT ILIKE '%pre-eclampsia%'
        AND ALL_DIAGNOSES NOT ILIKE '%preeclampsia%'
        AND ALL_DIAGNOSES NOT ILIKE '%eclampsia%'
        AND ALL_DIAGNOSES NOT ILIKE '%gestational hypertension%'
        AND ALL_DIAGNOSES NOT ILIKE '%pregnancy-induced hypertension%'
        AND ALL_DIAGNOSES NOT ILIKE '%pih%'
        AND ALL_DIAGNOSES NOT ILIKE '%complicating pregnancy%'
        AND age_in_years_float IS NOT NULL
        AND sex IN ('F', 'M')
        AND is_default_code_pair IS DISTINCT FROM TRUE
    GROUP BY
        age_group,
        sex
    ORDER BY
        MIN(age_in_years_float),
        sex
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_headline() -> pd.DataFrame:
    """N4.0 Distinct patients with any NCD, per year and overall."""
    return _run("q4_headline", f"""
    WITH labelled AS (
        SELECT
            unique_patient_id,
            visit_year,
            age_in_years_float,
            sex,
            county,
            ward,
            CASE
                WHEN diagnosis ILIKE '%hypertens%'
                    AND NOT (
                        diagnosis ILIKE ANY (
                            '%ocular%',
                            '%pulmonary%',
                            '%portal%',
                            '%intracranial%',
                            '%eclampsia%',
                            '%gestational%',
                            '%pregnan%',
                            '%puerper%',
                            '%pih%'
                        )
                    )
                THEN 'Hypertension'
                WHEN diagnosis ILIKE '%diabetes%'
                    AND NOT (
                        diagnosis ILIKE ANY ('%insipidus%', '%gestational%', '%in pregnancy%', '%without associated diabetes%')
                    )
                THEN 'Diabetes'
                WHEN diagnosis ILIKE ANY ('%chronic kidney%', '%chronic renal%', '%end-stage renal%', '%end stage renal%') THEN 'Chronic kidney disease'
                WHEN diagnosis ILIKE '%epilep%' THEN 'Epilepsy'
                WHEN diagnosis ILIKE ANY ('%heart failure%', '%cardiac failure%') THEN 'Heart failure'
                WHEN diagnosis ILIKE ANY ('%stroke%', '%cerebrovascular%') THEN 'Stroke / cerebrovascular disease'
                WHEN diagnosis ILIKE '%asthma%' THEN 'Asthma'
                WHEN diagnosis ILIKE ANY ('%chronic obstructive%', '%copd%') THEN 'COPD'
                WHEN diagnosis ILIKE ANY ('%malignant%', '%carcinoma%', '%cancer%', '%lymphoma%', '%leukaemia%', '%leukemia%', '%sarcoma%', '%myeloma%')
                    AND NOT (
                        diagnosis ILIKE ANY ('%benign%', '%in situ%')
                    )
                THEN 'Cancer'
                WHEN diagnosis ILIKE ANY ('%sickle cell%', '%sickle-cell%') AND diagnosis NOT ILIKE '%trait%' THEN 'Sickle cell disease'
            END AS condition_name
        FROM {TABLE}
        UNPIVOT (diagnosis FOR slot IN (diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4, diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8))
        WHERE
            is_default_code_pair IS DISTINCT FROM TRUE
    ), ncd_pt AS (
        SELECT DISTINCT
            unique_patient_id,
            visit_year
        FROM labelled
        WHERE
            condition_name IS NOT NULL
    ), all_pt AS (
        SELECT DISTINCT
            unique_patient_id,
            visit_year
        FROM {TABLE}
        WHERE
            is_default_code_pair IS DISTINCT FROM TRUE AND ALL_DIAGNOSES IS NOT NULL
    )
    SELECT
        COALESCE(TO_CHAR(a.visit_year), 'All years') AS period,
        COUNT(DISTINCT a.unique_patient_id) AS all_patients,
        COUNT(DISTINCT n.unique_patient_id) AS ncd_patients,
        ROUND(100.0 * COUNT(DISTINCT n.unique_patient_id) / COUNT(DISTINCT a.unique_patient_id), 1) AS pct_ncd
    FROM all_pt AS a
    LEFT JOIN ncd_pt AS n
        ON n.unique_patient_id = a.unique_patient_id AND n.visit_year = a.visit_year
    GROUP BY
        GROUPING SETS (
            (
                a.visit_year
            ),
            ()
        )
    ORDER BY
        a.visit_year
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_anaemia_sign() -> pd.DataFrame:
    """N4.A Share of each NCD's patients who also carry an anaemia code."""
    return _run("q4_anaemia_sign", f"""
    WITH labelled AS (
        SELECT
            unique_patient_id,
            visit_year,
            age_in_years_float,
            sex,
            county,
            ward,
            CASE
                WHEN diagnosis ILIKE '%hypertens%'
                    AND NOT (
                        diagnosis ILIKE ANY (
                            '%ocular%',
                            '%pulmonary%',
                            '%portal%',
                            '%intracranial%',
                            '%eclampsia%',
                            '%gestational%',
                            '%pregnan%',
                            '%puerper%',
                            '%pih%'
                        )
                    )
                THEN 'Hypertension'
                WHEN diagnosis ILIKE '%diabetes%'
                    AND NOT (
                        diagnosis ILIKE ANY ('%insipidus%', '%gestational%', '%in pregnancy%', '%without associated diabetes%')
                    )
                THEN 'Diabetes'
                WHEN diagnosis ILIKE ANY ('%chronic kidney%', '%chronic renal%', '%end-stage renal%', '%end stage renal%') THEN 'Chronic kidney disease'
                WHEN diagnosis ILIKE '%epilep%' THEN 'Epilepsy'
                WHEN diagnosis ILIKE ANY ('%heart failure%', '%cardiac failure%') THEN 'Heart failure'
                WHEN diagnosis ILIKE ANY ('%stroke%', '%cerebrovascular%') THEN 'Stroke / cerebrovascular disease'
                WHEN diagnosis ILIKE '%asthma%' THEN 'Asthma'
                WHEN diagnosis ILIKE ANY ('%chronic obstructive%', '%copd%') THEN 'COPD'
                WHEN diagnosis ILIKE ANY ('%malignant%', '%carcinoma%', '%cancer%', '%lymphoma%', '%leukaemia%', '%leukemia%', '%sarcoma%', '%myeloma%')
                    AND NOT (
                        diagnosis ILIKE ANY ('%benign%', '%in situ%')
                    )
                THEN 'Cancer'
                WHEN diagnosis ILIKE ANY ('%sickle cell%', '%sickle-cell%') AND diagnosis NOT ILIKE '%trait%' THEN 'Sickle cell disease'
            END AS condition_name
        FROM {TABLE}
        UNPIVOT (diagnosis FOR slot IN (diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4, diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8))
        WHERE
            is_default_code_pair IS DISTINCT FROM TRUE
    ), ncd_pt AS (
        SELECT DISTINCT
            condition_name,
            unique_patient_id
        FROM labelled
        WHERE
            condition_name IS NOT NULL
    ), anaemia AS (
        SELECT DISTINCT
            unique_patient_id
        FROM {TABLE}
        WHERE
            is_default_code_pair IS DISTINCT FROM TRUE AND ALL_DIAGNOSES ILIKE ANY ('%anaemia%', '%anemia%')
    )
    SELECT
        p.condition_name,
        COUNT(*) AS patients,
        COUNT(a.unique_patient_id) AS with_anaemia,
        ROUND(100.0 * COUNT(a.unique_patient_id) / COUNT(*), 1) AS pct_with_anaemia
    FROM ncd_pt AS p
    LEFT JOIN anaemia AS a
        ON a.unique_patient_id = p.unique_patient_id
    GROUP BY
        p.condition_name
    ORDER BY
        pct_with_anaemia DESC
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_diabetes_type() -> pd.DataFrame:
    """N4.D Diabetes type by age group (gestational and insipidus shown as excluded)."""
    return _run("q4_diabetes_type", f"""
    SELECT
        CASE
            WHEN diagnosis ILIKE '%insipidus%' THEN 'Excluded: diabetes insipidus'
            WHEN diagnosis ILIKE '%without associated diabetes%' THEN 'Excluded: hypoglycaemia without diabetes'
            WHEN diagnosis ILIKE ANY ('%gestational%', '%in pregnancy%') THEN 'Excluded: gestational'
            WHEN diagnosis ILIKE ANY ('%type 1%', '%type i %', '%insulin-dependent%', '%insulin dependent%', '%iddm%') THEN 'Type 1'
            WHEN diagnosis ILIKE ANY ('%type 2%', '%type ii%', '%non-insulin%', '%niddm%') THEN 'Type 2'
            ELSE 'Type not specified'
        END AS diabetes_type,
        DECODE(
            CASE
                WHEN age_in_years_float < 5 THEN 1
                WHEN age_in_years_float < 13 THEN 2
                WHEN age_in_years_float < 18 THEN 3
                WHEN age_in_years_float < 25 THEN 4
                WHEN age_in_years_float < 35 THEN 5
                WHEN age_in_years_float < 45 THEN 6
                WHEN age_in_years_float < 55 THEN 7
                ELSE 8
            END,
            1,
            'Toddler (0-4)',
            2,
            'Child (5-12)',
            3,
            'Adolescent (13-17)',
            4,
            'Youth (18-24)',
            5,
            'Young Adult (25-34)',
            6,
            'Adult (35-44)',
            7,
            'Middle Age (45-54)',
            8,
            '55+'
        ) AS age_group,
        COUNT(DISTINCT unique_patient_id) AS patients
    FROM {TABLE}
    UNPIVOT (diagnosis FOR slot IN (diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4, diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8))
    WHERE
        is_default_code_pair IS DISTINCT FROM TRUE AND age_in_years_float IS NOT NULL AND diagnosis ILIKE '%diabetes%'
    GROUP BY
        1,
        CASE
            WHEN age_in_years_float < 5 THEN 1
            WHEN age_in_years_float < 13 THEN 2
            WHEN age_in_years_float < 18 THEN 3
            WHEN age_in_years_float < 25 THEN 4
            WHEN age_in_years_float < 35 THEN 5
            WHEN age_in_years_float < 45 THEN 6
            WHEN age_in_years_float < 55 THEN 7
            ELSE 8
        END,
        2
    ORDER BY
        diabetes_type,
        CASE
            WHEN age_in_years_float < 5 THEN 1
            WHEN age_in_years_float < 13 THEN 2
            WHEN age_in_years_float < 18 THEN 3
            WHEN age_in_years_float < 25 THEN 4
            WHEN age_in_years_float < 35 THEN 5
            WHEN age_in_years_float < 45 THEN 6
            WHEN age_in_years_float < 55 THEN 7
            ELSE 8
        END
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_heart_failure() -> pd.DataFrame:
    """N4.HF Heart failure by age group and sex with the peripartum test."""
    return _run("q4_heart_failure", f"""
    SELECT
        DECODE(
            CASE
                WHEN age_in_years_float < 5 THEN 1
                WHEN age_in_years_float < 13 THEN 2
                WHEN age_in_years_float < 18 THEN 3
                WHEN age_in_years_float < 25 THEN 4
                WHEN age_in_years_float < 35 THEN 5
                WHEN age_in_years_float < 45 THEN 6
                WHEN age_in_years_float < 55 THEN 7
                ELSE 8
            END,
            1,
            'Toddler (0-4)',
            2,
            'Child (5-12)',
            3,
            'Adolescent (13-17)',
            4,
            'Youth (18-24)',
            5,
            'Young Adult (25-34)',
            6,
            'Adult (35-44)',
            7,
            'Middle Age (45-54)',
            8,
            '55+'
        ) AS age_group,
        sex,
        COUNT(DISTINCT unique_patient_id) AS patients,
        COUNT(
            DISTINCT IFF(
                ALL_DIAGNOSES ILIKE ANY ('%pregnan%', '%peripartum%', '%postpartum%', '%puerper%', '%antenatal%'),
                unique_patient_id,
                NULL
            )
        ) AS with_pregnancy_term,
        COUNT(DISTINCT IFF(ALL_DIAGNOSES ILIKE '%cardiomyopath%', unique_patient_id, NULL)) AS with_cardiomyopathy,
        COUNT(DISTINCT IFF(ALL_DIAGNOSES ILIKE '%hypertens%', unique_patient_id, NULL)) AS with_hypertension
    FROM {TABLE}
    WHERE
        is_default_code_pair IS DISTINCT FROM TRUE
        AND ALL_DIAGNOSES IS NOT NULL
        AND age_in_years_float IS NOT NULL
        AND ALL_DIAGNOSES ILIKE ANY ('%heart failure%', '%cardiac failure%')
    GROUP BY
        CASE
            WHEN age_in_years_float < 5 THEN 1
            WHEN age_in_years_float < 13 THEN 2
            WHEN age_in_years_float < 18 THEN 3
            WHEN age_in_years_float < 25 THEN 4
            WHEN age_in_years_float < 35 THEN 5
            WHEN age_in_years_float < 45 THEN 6
            WHEN age_in_years_float < 55 THEN 7
            ELSE 8
        END,
        sex
    ORDER BY
        CASE
            WHEN age_in_years_float < 5 THEN 1
            WHEN age_in_years_float < 13 THEN 2
            WHEN age_in_years_float < 18 THEN 3
            WHEN age_in_years_float < 25 THEN 4
            WHEN age_in_years_float < 35 THEN 5
            WHEN age_in_years_float < 45 THEN 6
            WHEN age_in_years_float < 55 THEN 7
            ELSE 8
        END,
        sex
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_stroke() -> pd.DataFrame:
    """N4.S Stroke by age band and overlap with sickle cell disease."""
    return _run("q4_stroke", f"""
    WITH labelled AS (
        SELECT
            unique_patient_id,
            visit_year,
            age_in_years_float,
            sex,
            county,
            ward,
            CASE
                WHEN diagnosis ILIKE '%hypertens%'
                    AND NOT (
                        diagnosis ILIKE ANY (
                            '%ocular%',
                            '%pulmonary%',
                            '%portal%',
                            '%intracranial%',
                            '%eclampsia%',
                            '%gestational%',
                            '%pregnan%',
                            '%puerper%',
                            '%pih%'
                        )
                    )
                THEN 'Hypertension'
                WHEN diagnosis ILIKE '%diabetes%'
                    AND NOT (
                        diagnosis ILIKE ANY ('%insipidus%', '%gestational%', '%in pregnancy%', '%without associated diabetes%')
                    )
                THEN 'Diabetes'
                WHEN diagnosis ILIKE ANY ('%chronic kidney%', '%chronic renal%', '%end-stage renal%', '%end stage renal%') THEN 'Chronic kidney disease'
                WHEN diagnosis ILIKE '%epilep%' THEN 'Epilepsy'
                WHEN diagnosis ILIKE ANY ('%heart failure%', '%cardiac failure%') THEN 'Heart failure'
                WHEN diagnosis ILIKE ANY ('%stroke%', '%cerebrovascular%') THEN 'Stroke / cerebrovascular disease'
                WHEN diagnosis ILIKE '%asthma%' THEN 'Asthma'
                WHEN diagnosis ILIKE ANY ('%chronic obstructive%', '%copd%') THEN 'COPD'
                WHEN diagnosis ILIKE ANY ('%malignant%', '%carcinoma%', '%cancer%', '%lymphoma%', '%leukaemia%', '%leukemia%', '%sarcoma%', '%myeloma%')
                    AND NOT (
                        diagnosis ILIKE ANY ('%benign%', '%in situ%')
                    )
                THEN 'Cancer'
                WHEN diagnosis ILIKE ANY ('%sickle cell%', '%sickle-cell%') AND diagnosis NOT ILIKE '%trait%' THEN 'Sickle cell disease'
            END AS condition_name
        FROM {TABLE}
        UNPIVOT (diagnosis FOR slot IN (diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4, diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8))
        WHERE
            is_default_code_pair IS DISTINCT FROM TRUE AND age_in_years_float IS NOT NULL
    ), pt AS (
        SELECT
            unique_patient_id,
            MAX(age_in_years_float) AS age,
            MAX(IFF(condition_name = 'Stroke / cerebrovascular disease', 1, 0)) AS stroke,
            MAX(IFF(condition_name = 'Sickle cell disease', 1, 0)) AS sickle
        FROM labelled
        GROUP BY
            unique_patient_id
    )
    SELECT
        CASE
            WHEN age < 5 THEN 'Toddler (0-4)'
            WHEN age < 18 THEN 'Child / adolescent (5-17)'
            WHEN age < 45 THEN 'Adult (18-44)'
            ELSE '45+'
        END AS age_band,
        COUNT(*) AS stroke_patients,
        COUNT_IF(sickle = 1) AS also_sickle_cell,
        ROUND(100.0 * COUNT_IF(sickle = 1) / COUNT(*), 1) AS pct_sickle
    FROM pt
    WHERE
        stroke = 1
    GROUP BY
        1
    ORDER BY
        MIN(age)
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_ward_patients() -> pd.DataFrame:
    """N4.M1 NCD patients per ward (Nandi residents, 2024-2025) with ward geolocation: feeds the "Where they come from" map."""
    return _run("q4_ward_patients", f"""
    WITH dx AS (
        SELECT
            unique_patient_id,
            ward,
            sub_county,
            CASE
                WHEN diagnosis ILIKE '%hypertens%'
                    AND NOT (diagnosis ILIKE ANY ('%ocular%', '%pulmonary%', '%portal%', '%intracranial%', '%eclampsia%', '%gestational%', '%pregnan%', '%puerper%', '%pih%')) THEN 'Hypertension'
                WHEN diagnosis ILIKE '%diabetes%'
                    AND NOT (diagnosis ILIKE ANY ('%insipidus%', '%gestational%', '%in pregnancy%', '%without associated diabetes%')) THEN 'Diabetes'
                WHEN diagnosis ILIKE ANY ('%chronic kidney%', '%chronic renal%', '%end-stage renal%', '%end stage renal%') THEN 'Chronic kidney disease'
                WHEN diagnosis ILIKE '%epilep%' THEN 'Epilepsy'
                WHEN diagnosis ILIKE ANY ('%heart failure%', '%cardiac failure%') THEN 'Heart failure'
                WHEN diagnosis ILIKE ANY ('%stroke%', '%cerebrovascular%') THEN 'Stroke / cerebrovascular disease'
                WHEN diagnosis ILIKE '%asthma%' THEN 'Asthma'
                WHEN diagnosis ILIKE ANY ('%chronic obstructive%', '%copd%') THEN 'COPD'
                WHEN diagnosis ILIKE ANY ('%malignant%', '%carcinoma%', '%cancer%', '%lymphoma%', '%leukaemia%', '%leukemia%', '%sarcoma%', '%myeloma%')
                    AND NOT (diagnosis ILIKE ANY ('%benign%', '%in situ%')) THEN 'Cancer'
                WHEN diagnosis ILIKE ANY ('%sickle cell%', '%sickle-cell%')
                    AND diagnosis NOT ILIKE '%trait%' THEN 'Sickle cell disease'
            END AS condition_name
        FROM {TABLE}
        UNPIVOT (
            diagnosis FOR slot IN (
                diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4,
                diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
            )
        )
        WHERE is_default_code_pair IS DISTINCT FROM TRUE
          AND county = 'Nandi'
          AND ward IS NOT NULL
          AND visit_year IN (2024, 2025)
    ),
    ward_counts AS (
        SELECT
            ward,
            sub_county,
            COALESCE(condition_name, 'All NCDs') AS condition_name,
            COUNT(DISTINCT unique_patient_id) AS patients
        FROM dx
        WHERE condition_name IS NOT NULL
        GROUP BY GROUPING SETS ((ward, sub_county, condition_name), (ward, sub_county))
    )
    SELECT
        c.ward,
        c.sub_county,
        c.condition_name,
        c.patients,
        g.latitude,
        g.longitude,
        g.boundary_geojson
    FROM ward_counts c
    LEFT JOIN {GEO_TABLE} g
           ON g.geo_level = 'WARD'
          AND g.geocode_status = 'FOUND'
          AND g.county ILIKE 'Nandi'
          AND UPPER(TRIM(g.geo_name))    = UPPER(TRIM(c.ward))
          AND UPPER(TRIM(g.parent_name)) = UPPER(TRIM(c.sub_county))
    ORDER BY c.condition_name, c.patients DESC
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_ward_recurrent() -> pd.DataFrame:
    """N4.M2 Recurrent NCD patients (seen in both 2024 and 2025) per ward with ward geolocation: feeds the recurrence map."""
    return _run("q4_ward_recurrent", f"""
    WITH dx AS (
        SELECT
            unique_patient_id,
            visit_year,
            ward,
            sub_county,
            CASE
                WHEN diagnosis ILIKE '%hypertens%'
                    AND NOT (diagnosis ILIKE ANY ('%ocular%', '%pulmonary%', '%portal%', '%intracranial%', '%eclampsia%', '%gestational%', '%pregnan%', '%puerper%', '%pih%')) THEN 'Hypertension'
                WHEN diagnosis ILIKE '%diabetes%'
                    AND NOT (diagnosis ILIKE ANY ('%insipidus%', '%gestational%', '%in pregnancy%', '%without associated diabetes%')) THEN 'Diabetes'
                WHEN diagnosis ILIKE ANY ('%chronic kidney%', '%chronic renal%', '%end-stage renal%', '%end stage renal%') THEN 'Chronic kidney disease'
                WHEN diagnosis ILIKE '%epilep%' THEN 'Epilepsy'
                WHEN diagnosis ILIKE ANY ('%heart failure%', '%cardiac failure%') THEN 'Heart failure'
                WHEN diagnosis ILIKE ANY ('%stroke%', '%cerebrovascular%') THEN 'Stroke / cerebrovascular disease'
                WHEN diagnosis ILIKE '%asthma%' THEN 'Asthma'
                WHEN diagnosis ILIKE ANY ('%chronic obstructive%', '%copd%') THEN 'COPD'
                WHEN diagnosis ILIKE ANY ('%malignant%', '%carcinoma%', '%cancer%', '%lymphoma%', '%leukaemia%', '%leukemia%', '%sarcoma%', '%myeloma%')
                    AND NOT (diagnosis ILIKE ANY ('%benign%', '%in situ%')) THEN 'Cancer'
                WHEN diagnosis ILIKE ANY ('%sickle cell%', '%sickle-cell%')
                    AND diagnosis NOT ILIKE '%trait%' THEN 'Sickle cell disease'
            END AS condition_name
        FROM {TABLE}
        UNPIVOT (
            diagnosis FOR slot IN (
                diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4,
                diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
            )
        )
        WHERE is_default_code_pair IS DISTINCT FROM TRUE
          AND county = 'Nandi'
          AND ward IS NOT NULL
          AND visit_year IN (2024, 2025)
    ),
    per_patient AS (
        SELECT
            unique_patient_id,
            MODE(ward)       AS ward,
            MODE(sub_county) AS sub_county,
            COUNT(DISTINCT visit_year) AS years_seen
        FROM dx
        WHERE condition_name IS NOT NULL
        GROUP BY unique_patient_id
    ),
    ward_counts AS (
        SELECT
            ward,
            sub_county,
            COUNT(*) AS ncd_patients,
            COUNT_IF(years_seen = 2) AS seen_both_years,
            ROUND(100.0 * COUNT_IF(years_seen = 2) / COUNT(*), 1) AS pct_seen_both_years
        FROM per_patient
        GROUP BY ward, sub_county
    )
    SELECT
        c.ward,
        c.sub_county,
        c.ncd_patients,
        c.seen_both_years,
        c.pct_seen_both_years,
        g.latitude,
        g.longitude,
        g.boundary_geojson
    FROM ward_counts c
    LEFT JOIN {GEO_TABLE} g
           ON g.geo_level = 'WARD'
          AND g.geocode_status = 'FOUND'
          AND g.county ILIKE 'Nandi'
          AND UPPER(TRIM(g.geo_name))    = UPPER(TRIM(c.ward))
          AND UPPER(TRIM(g.parent_name)) = UPPER(TRIM(c.sub_county))
    ORDER BY c.ncd_patients DESC
    """)

@st.cache_data(ttl=3600, show_spinner=False)
def q4_condition_return() -> pd.DataFrame:
    """N4.R1 Share of each condition's patients seen in both 2024 and 2025, overall and by age band."""
    return _run("q4_condition_return", f"""
    WITH dx AS (
        SELECT
            unique_patient_id,
            pno,
            visit_year,
            age_in_years_float,
            CASE
                WHEN diagnosis ILIKE '%hypertens%'
                    AND NOT (diagnosis ILIKE ANY ('%ocular%', '%pulmonary%', '%portal%', '%intracranial%', '%eclampsia%', '%gestational%', '%pregnan%', '%puerper%', '%pih%')) THEN 'Hypertension'
                WHEN diagnosis ILIKE '%diabetes%'
                    AND NOT (diagnosis ILIKE ANY ('%insipidus%', '%gestational%', '%in pregnancy%', '%without associated diabetes%')) THEN 'Diabetes'
                WHEN diagnosis ILIKE ANY ('%chronic kidney%', '%chronic renal%', '%end-stage renal%', '%end stage renal%') THEN 'Chronic kidney disease'
                WHEN diagnosis ILIKE '%epilep%' THEN 'Epilepsy'
                WHEN diagnosis ILIKE ANY ('%heart failure%', '%cardiac failure%') THEN 'Heart failure'
                WHEN diagnosis ILIKE ANY ('%stroke%', '%cerebrovascular%') THEN 'Stroke / cerebrovascular disease'
                WHEN diagnosis ILIKE '%asthma%' THEN 'Asthma'
                WHEN diagnosis ILIKE ANY ('%chronic obstructive%', '%copd%') THEN 'COPD'
                WHEN diagnosis ILIKE ANY ('%malignant%', '%carcinoma%', '%cancer%', '%lymphoma%', '%leukaemia%', '%leukemia%', '%sarcoma%', '%myeloma%')
                    AND NOT (diagnosis ILIKE ANY ('%benign%', '%in situ%')) THEN 'Cancer'
                WHEN diagnosis ILIKE ANY ('%sickle cell%', '%sickle-cell%')
                    AND diagnosis NOT ILIKE '%trait%' THEN 'Sickle cell disease'
            END AS condition_name
        FROM {TABLE}
        UNPIVOT (
            diagnosis FOR slot IN (
                diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4,
                diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
            )
        )
        WHERE is_default_code_pair IS DISTINCT FROM TRUE
          AND age_in_years_float IS NOT NULL
          AND visit_year IN (2024, 2025)
    ),
    per_patient AS (
        SELECT
            condition_name,
            unique_patient_id,
            COUNT(DISTINCT visit_year) AS years_seen,
            COUNT(DISTINCT pno) AS visits,
            CASE
                WHEN MAX(age_in_years_float) < 18 THEN 'Under 18'
                WHEN MAX(age_in_years_float) < 55 THEN '18-54'
                ELSE '55+'
            END AS age_band
        FROM dx
        WHERE condition_name IS NOT NULL
        GROUP BY condition_name, unique_patient_id
    )
    SELECT
        condition_name,
        COALESCE(age_band, 'All ages') AS age_band,
        COUNT(*) AS patients,
        COUNT_IF(years_seen = 2) AS seen_both_years,
        ROUND(100.0 * COUNT_IF(years_seen = 2) / COUNT(*), 1) AS pct_seen_both_years,
        ROUND(AVG(visits), 2) AS avg_visits_with_condition
    FROM per_patient
    GROUP BY GROUPING SETS ((condition_name, age_band), (condition_name))
    ORDER BY condition_name, age_band
    """)


# =====================================================================================================
# TAB 5 · PSYCHIATRIC CONDITIONS (q5_*). Source: mental_health_queries_v3.sql (Q5.1 to Q5.6).
# Definition: ICD-11 chapter 6, counted by patient, default code pair excluded. One row per visit with a pipe-joined
# string of its ICD-11 codes and one TRUE/FALSE flag per condition, UNPIVOTed into condition rows ("Any mental health"
# is just another flag). Schizophrenia 6A20 | Other psychotic 6A21-6A25, 6A2Y, 6A2Z, 6E61 | Bipolar 6A6 | Depression 6A7 |
# Anxiety 6B0, 6E63 | Trauma & stress-related 6B4 | Alcohol & substance use 6C4 | Dementia 6D8.
# Excluded: 6E20 perinatal code (449 patients, unreliable), autism/ADHD, dissociative, bodily distress.
# NEW queries (not in the earlier notes): q5_wards (Q5.4 ward map), q5_returning (Q5.5), q5_oddities (Q5.6).
# =====================================================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def q5_yearly() -> pd.DataFrame:
    """Q5.1 Patients per condition per year, with the all-patient denominator (also gives the headline and KPIs)"""
    return _run("q5_yearly", f"""
    WITH visits AS (
        SELECT
            unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES,
            '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
        FROM {TABLE}
        WHERE is_default_code_pair IS DISTINCT FROM TRUE
    ),
    flags AS (
        SELECT
            *,
            codes LIKE '%|6A20%'                                                                                         AS "Schizophrenia",
            codes LIKE ANY ('%|6A21%', '%|6A22%', '%|6A23%', '%|6A24%', '%|6A25%', '%|6A2Y%', '%|6A2Z%', '%|6E61%')     AS "Other psychotic disorders",
            codes LIKE '%|6A6%'                                                                                          AS "Bipolar disorder",
            codes LIKE '%|6A7%'                                                                                          AS "Depression",
            codes LIKE ANY ('%|6B0%', '%|6E63%')                                                                         AS "Anxiety",
            codes LIKE '%|6B4%'                                                                                          AS "Trauma & stress-related",
            codes LIKE '%|6C4%'                                                                                          AS "Alcohol & substance use",
            codes LIKE '%|6D8%'                                                                                          AS "Dementia",
            codes LIKE ANY ('%|6A2%', '%|6A6%', '%|6A7%', '%|6B0%', '%|6B4%', '%|6C4%', '%|6D8%', '%|6E61%', '%|6E63%') AS "Any mental health"
        FROM visits
    ),
    dx AS (
        SELECT *
        FROM flags
        UNPIVOT (is_match FOR condition_name IN ("Schizophrenia", "Other psychotic disorders", "Bipolar disorder", "Depression", "Anxiety", "Trauma & stress-related", "Alcohol & substance use", "Dementia", "Any mental health"))
        WHERE is_match
    ),
    all_pt AS (
        SELECT
            COALESCE(TO_CHAR(visit_year), 'All years') AS period,
            COUNT(DISTINCT unique_patient_id)          AS all_patients
        FROM {TABLE}
        WHERE is_default_code_pair IS DISTINCT FROM TRUE
          AND ALL_DIAGNOSES IS NOT NULL
        GROUP BY GROUPING SETS ((visit_year), ())
    ),
    mh AS (
        SELECT
            COALESCE(TO_CHAR(visit_year), 'All years') AS period,
            condition_name,
            COUNT(DISTINCT unique_patient_id)          AS patients
        FROM dx
        GROUP BY GROUPING SETS ((visit_year, condition_name), (condition_name))
    )
    SELECT
        m.period,
        m.condition_name,
        m.patients,
        a.all_patients,
        ROUND(100.0 * m.patients / a.all_patients, 3) AS pct_of_patients
    FROM mh m
    JOIN all_pt a ON a.period = m.period
    ORDER BY m.condition_name, m.period
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q5_age_sex() -> pd.DataFrame:
    """Q5.2 Patients per condition by age group at first diagnosis, sex and origin (all years)"""
    return _run("q5_age_sex", f"""
    WITH visits AS (
        SELECT
            unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES,
            '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
        FROM {TABLE}
        WHERE is_default_code_pair IS DISTINCT FROM TRUE
          AND age_in_years_float IS NOT NULL
    ),
    flags AS (
        SELECT
            *,
            codes LIKE '%|6A20%'                                                                                         AS "Schizophrenia",
            codes LIKE ANY ('%|6A21%', '%|6A22%', '%|6A23%', '%|6A24%', '%|6A25%', '%|6A2Y%', '%|6A2Z%', '%|6E61%')     AS "Other psychotic disorders",
            codes LIKE '%|6A6%'                                                                                          AS "Bipolar disorder",
            codes LIKE '%|6A7%'                                                                                          AS "Depression",
            codes LIKE ANY ('%|6B0%', '%|6E63%')                                                                         AS "Anxiety",
            codes LIKE '%|6B4%'                                                                                          AS "Trauma & stress-related",
            codes LIKE '%|6C4%'                                                                                          AS "Alcohol & substance use",
            codes LIKE '%|6D8%'                                                                                          AS "Dementia",
            codes LIKE ANY ('%|6A2%', '%|6A6%', '%|6A7%', '%|6B0%', '%|6B4%', '%|6C4%', '%|6D8%', '%|6E61%', '%|6E63%') AS "Any mental health"
        FROM visits
    ),
    dx AS (
        SELECT *
        FROM flags
        UNPIVOT (is_match FOR condition_name IN ("Schizophrenia", "Other psychotic disorders", "Bipolar disorder", "Depression", "Anxiety", "Trauma & stress-related", "Alcohol & substance use", "Dementia", "Any mental health"))
        WHERE is_match
    ),
    first_dx AS (
        SELECT
            condition_name,
            unique_patient_id,
            sex,
            county,
            age_in_years_float,
            ROW_NUMBER() OVER (PARTITION BY condition_name, unique_patient_id ORDER BY visit_year, age_in_years_float) AS rn
        FROM dx
    )
    SELECT
        condition_name,
        CASE
            WHEN age_in_years_float < 5  THEN 'Toddler (0-4)'
            WHEN age_in_years_float < 13 THEN 'Child (5-12)'
            WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
            WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
            WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
            WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
            WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
            ELSE '55+'
        END AS age_group_at_first_diagnosis,
        sex,
        COUNT(DISTINCT unique_patient_id)                  AS patients,
        COUNT_IF(county = 'Nandi')                         AS from_nandi,
        COUNT_IF(county <> 'Nandi' AND county IS NOT NULL) AS from_outside_nandi
    FROM first_dx
    WHERE rn = 1
    GROUP BY condition_name, age_group_at_first_diagnosis, sex
    ORDER BY condition_name, MIN(age_in_years_float), sex
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q5_comorbidity() -> pd.DataFrame:
    """Q5.3 Mental health condition x physical NCD, patient level (all years). FIXED: with_any_ncd now ignores patients with no NCD"""
    return _run("q5_comorbidity", f"""
    WITH slots AS (
        SELECT
            o.unique_patient_id,
            f.value:code::string AS code,
            f.value:text::string AS txt
        FROM {TABLE} o,
             LATERAL FLATTEN(INPUT => ARRAY_CONSTRUCT(OBJECT_CONSTRUCT('text', o.diagnosis_1, 'code', o.diagnosis_1_icd11), OBJECT_CONSTRUCT('text', o.diagnosis_2, 'code', o.diagnosis_2_icd11), OBJECT_CONSTRUCT('text', o.diagnosis_3, 'code', o.diagnosis_3_icd11), OBJECT_CONSTRUCT('text', o.diagnosis_4, 'code', o.diagnosis_4_icd11), OBJECT_CONSTRUCT('text', o.diagnosis_5, 'code', o.diagnosis_5_icd11), OBJECT_CONSTRUCT('text', o.diagnosis_6, 'code', o.diagnosis_6_icd11), OBJECT_CONSTRUCT('text', o.diagnosis_7, 'code', o.diagnosis_7_icd11), OBJECT_CONSTRUCT('text', o.diagnosis_8, 'code', o.diagnosis_8_icd11))) f
        WHERE o.is_default_code_pair IS DISTINCT FROM TRUE
    ),
    mh AS (
        SELECT DISTINCT
            unique_patient_id,
            CASE
                WHEN code LIKE '6A20%' THEN 'Schizophrenia'
                WHEN code LIKE ANY ('6A2%', '6E61%') THEN 'Other psychotic disorders'
                WHEN code LIKE '6A6%' THEN 'Bipolar disorder'
                WHEN code LIKE '6A7%' THEN 'Depression'
                WHEN code LIKE ANY ('6B0%', '6E63%') THEN 'Anxiety'
                WHEN code LIKE '6B4%' THEN 'Trauma & stress-related'
                WHEN code LIKE '6C4%' THEN 'Alcohol & substance use'
                WHEN code LIKE '6D8%' THEN 'Dementia'
            END AS condition_name
        FROM slots
        WHERE code IS NOT NULL
    ),
    ncd AS (
        SELECT DISTINCT
            unique_patient_id,
            CASE
                    WHEN txt ILIKE '%hypertens%'
                        AND NOT (txt ILIKE ANY ('%ocular%', '%pulmonary%', '%portal%', '%intracranial%', '%eclampsia%', '%gestational%', '%pregnan%', '%puerper%', '%pih%')) THEN 'Hypertension'
                    WHEN txt ILIKE '%diabetes%'
                        AND NOT (txt ILIKE ANY ('%insipidus%', '%gestational%', '%in pregnancy%', '%without associated diabetes%')) THEN 'Diabetes'
                    WHEN txt ILIKE ANY ('%chronic kidney%', '%chronic renal%', '%end-stage renal%', '%end stage renal%') THEN 'Chronic kidney disease'
                    WHEN txt ILIKE '%epilep%' THEN 'Epilepsy'
                    WHEN txt ILIKE ANY ('%heart failure%', '%cardiac failure%') THEN 'Heart failure'
                    WHEN txt ILIKE ANY ('%stroke%', '%cerebrovascular%') THEN 'Stroke'
                    WHEN txt ILIKE '%asthma%' THEN 'Asthma'
                END AS ncd_name
        FROM slots
        WHERE txt IS NOT NULL
    )
    SELECT
        m.condition_name,
        COUNT(DISTINCT m.unique_patient_id)                                                   AS patients,
        COUNT(DISTINCT IFF(n.ncd_name = 'Hypertension',          m.unique_patient_id, NULL))  AS with_hypertension,
        COUNT(DISTINCT IFF(n.ncd_name = 'Diabetes',              m.unique_patient_id, NULL))  AS with_diabetes,
        COUNT(DISTINCT IFF(n.ncd_name = 'Epilepsy',              m.unique_patient_id, NULL))  AS with_epilepsy,
        COUNT(DISTINCT IFF(n.ncd_name = 'Heart failure',         m.unique_patient_id, NULL))  AS with_heart_failure,
        COUNT(DISTINCT IFF(n.ncd_name = 'Chronic kidney disease',m.unique_patient_id, NULL))  AS with_ckd,
        COUNT(DISTINCT IFF(n.ncd_name = 'Asthma',                m.unique_patient_id, NULL))  AS with_asthma,
        COUNT(DISTINCT IFF(n.ncd_name = 'Stroke',                m.unique_patient_id, NULL))  AS with_stroke,
        COUNT(DISTINCT IFF(n.ncd_name IS NOT NULL, m.unique_patient_id, NULL)) AS with_any_ncd
    FROM mh m
    LEFT JOIN ncd n ON n.unique_patient_id = m.unique_patient_id
    WHERE m.condition_name IS NOT NULL
    GROUP BY m.condition_name
    ORDER BY patients DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q5_wards() -> pd.DataFrame:
    """Q5.4 Patients per ward and condition (Nandi residents, 2024-2025), with ward geolocation for the map"""
    return _run("q5_wards", f"""
    WITH visits AS (
        SELECT
            unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES,
            '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
        FROM {TABLE}
        WHERE is_default_code_pair IS DISTINCT FROM TRUE
          AND county = 'Nandi'
          AND ward IS NOT NULL
          AND visit_year IN (2024, 2025)
    ),
    flags AS (
        SELECT
            *,
            codes LIKE '%|6A20%'                                                                                         AS "Schizophrenia",
            codes LIKE ANY ('%|6A21%', '%|6A22%', '%|6A23%', '%|6A24%', '%|6A25%', '%|6A2Y%', '%|6A2Z%', '%|6E61%')     AS "Other psychotic disorders",
            codes LIKE '%|6A6%'                                                                                          AS "Bipolar disorder",
            codes LIKE '%|6A7%'                                                                                          AS "Depression",
            codes LIKE ANY ('%|6B0%', '%|6E63%')                                                                         AS "Anxiety",
            codes LIKE '%|6B4%'                                                                                          AS "Trauma & stress-related",
            codes LIKE '%|6C4%'                                                                                          AS "Alcohol & substance use",
            codes LIKE '%|6D8%'                                                                                          AS "Dementia",
            codes LIKE ANY ('%|6A2%', '%|6A6%', '%|6A7%', '%|6B0%', '%|6B4%', '%|6C4%', '%|6D8%', '%|6E61%', '%|6E63%') AS "Any mental health"
        FROM visits
    ),
    dx AS (
        SELECT *
        FROM flags
        UNPIVOT (is_match FOR condition_name IN ("Schizophrenia", "Other psychotic disorders", "Bipolar disorder", "Depression", "Anxiety", "Trauma & stress-related", "Alcohol & substance use", "Dementia", "Any mental health"))
        WHERE is_match
    ),
    ward_counts AS (
        SELECT
            ward,
            sub_county,
            condition_name,
            COUNT(DISTINCT unique_patient_id) AS patients
        FROM dx
        GROUP BY ward, sub_county, condition_name
    )
    SELECT
        c.ward,
        c.sub_county,
        c.condition_name,
        c.patients,
        g.latitude,
        g.longitude,
        g.boundary_geojson
    FROM ward_counts c
    LEFT JOIN {GEO_TABLE} g
           ON g.geo_level = 'WARD'
          AND g.geocode_status = 'FOUND'
          AND g.county ILIKE 'Nandi'
          AND UPPER(TRIM(g.geo_name))    = UPPER(TRIM(c.ward))
          AND UPPER(TRIM(g.parent_name)) = UPPER(TRIM(c.sub_county))
    ORDER BY c.condition_name, c.patients DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q5_returning() -> pd.DataFrame:
    """Q5.5 Patients seen in both 2024 and 2025, per condition and age band (the view adds the all-ages row and the % )"""
    return _run("q5_returning", f"""
    WITH visits AS (
        SELECT
            unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES,
            '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
        FROM {TABLE}
        WHERE is_default_code_pair IS DISTINCT FROM TRUE
          AND age_in_years_float IS NOT NULL
          AND visit_year IN (2024, 2025)
    ),
    flags AS (
        SELECT
            *,
            codes LIKE '%|6A20%'                                                                                         AS "Schizophrenia",
            codes LIKE ANY ('%|6A21%', '%|6A22%', '%|6A23%', '%|6A24%', '%|6A25%', '%|6A2Y%', '%|6A2Z%', '%|6E61%')     AS "Other psychotic disorders",
            codes LIKE '%|6A6%'                                                                                          AS "Bipolar disorder",
            codes LIKE '%|6A7%'                                                                                          AS "Depression",
            codes LIKE ANY ('%|6B0%', '%|6E63%')                                                                         AS "Anxiety",
            codes LIKE '%|6B4%'                                                                                          AS "Trauma & stress-related",
            codes LIKE '%|6C4%'                                                                                          AS "Alcohol & substance use",
            codes LIKE '%|6D8%'                                                                                          AS "Dementia",
            codes LIKE ANY ('%|6A2%', '%|6A6%', '%|6A7%', '%|6B0%', '%|6B4%', '%|6C4%', '%|6D8%', '%|6E61%', '%|6E63%') AS "Any mental health"
        FROM visits
    ),
    dx AS (
        SELECT *
        FROM flags
        UNPIVOT (is_match FOR condition_name IN ("Schizophrenia", "Other psychotic disorders", "Bipolar disorder", "Depression", "Anxiety", "Trauma & stress-related", "Alcohol & substance use", "Dementia", "Any mental health"))
        WHERE is_match
    ),
    per_patient AS (
        SELECT
            condition_name,
            unique_patient_id,
            COUNT(DISTINCT visit_year) AS years_seen,
            COUNT(DISTINCT pno)        AS visits,
            CASE
                WHEN MAX(age_in_years_float) < 18 THEN 'Under 18'
                WHEN MAX(age_in_years_float) < 55 THEN '18-54'
                ELSE '55+'
            END AS age_band
        FROM dx
        GROUP BY condition_name, unique_patient_id
    )
    SELECT
        condition_name,
        age_band,
        COUNT(*)                         AS patients,
        COUNT_IF(years_seen = 2)         AS seen_both_years,
        SUM(visits)                      AS total_visits
    FROM per_patient
    GROUP BY condition_name, age_band
    ORDER BY condition_name, age_band
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q5_oddities() -> pd.DataFrame:
    """Q5.6 Patterns that do not fit: the full diagnosis text and codes behind each oddity"""
    return _run("q5_oddities", f"""
    WITH visits AS (
        SELECT
            unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES,
            '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
        FROM {TABLE}
        WHERE is_default_code_pair IS DISTINCT FROM TRUE
          AND age_in_years_float IS NOT NULL
    ),
    flags AS (
        SELECT
            *,
            codes LIKE '%|6A20%'                                                                                         AS "Schizophrenia",
            codes LIKE ANY ('%|6A21%', '%|6A22%', '%|6A23%', '%|6A24%', '%|6A25%', '%|6A2Y%', '%|6A2Z%', '%|6E61%')     AS "Other psychotic disorders",
            codes LIKE '%|6A6%'                                                                                          AS "Bipolar disorder",
            codes LIKE '%|6A7%'                                                                                          AS "Depression",
            codes LIKE ANY ('%|6B0%', '%|6E63%')                                                                         AS "Anxiety",
            codes LIKE '%|6B4%'                                                                                          AS "Trauma & stress-related",
            codes LIKE '%|6C4%'                                                                                          AS "Alcohol & substance use",
            codes LIKE '%|6D8%'                                                                                          AS "Dementia",
            codes LIKE ANY ('%|6A2%', '%|6A6%', '%|6A7%', '%|6B0%', '%|6B4%', '%|6C4%', '%|6D8%', '%|6E61%', '%|6E63%') AS "Any mental health"
        FROM visits
    )
    SELECT
        CASE
            WHEN "Schizophrenia"           AND age_in_years_float < 13 THEN 'Schizophrenia under 13'
            WHEN "Alcohol & substance use" AND age_in_years_float < 18 THEN 'Alcohol / substance, under 18'
            WHEN "Alcohol & substance use" AND sex = 'F'               THEN 'Alcohol / substance, women'
            WHEN "Dementia"                AND age_in_years_float < 55 THEN 'Dementia under 55'
        END AS oddity,
        sex,
        CASE WHEN age_in_years_float < 13 THEN 'Under 13' WHEN age_in_years_float < 18 THEN '13-17' WHEN age_in_years_float < 55 THEN '18-54' ELSE '55+' END AS age_band,
        ALL_DIAGNOSES                    AS diagnosis_text,
        LTRIM(codes, '|')                AS icd11_codes,
        COUNT(DISTINCT unique_patient_id) AS patients
    FROM flags
    WHERE ("Schizophrenia"           AND age_in_years_float < 13)
       OR ("Alcohol & substance use" AND (age_in_years_float < 18 OR sex = 'F'))
       OR ("Dementia"                AND age_in_years_float < 55)
    GROUP BY ALL
    ORDER BY oddity, patients DESC
    LIMIT 80
    """)



# =====================================================================================================
# TAB 6 · RESPIRATORY. Source of truth: respiratory_queries.sql (Q6.1 to Q6.5 and Q6.7; Q6.6 aspergillosis is a Data quality
# check and is not run here). ICD-11 chapter 12 (CA, CB), counted by patient, default code pair excluded. Burdens are mapped by
# code, because this dataset reuses codes across texts (CA09.0 tonsillitis and pharyngitis, CA08.Z upper respiratory infection,
# CA08.0 allergic rhinitis): Pneumonia CA40 | Upper respiratory infection CA07, CA08.Z, CA01, CA04, CA05 | Acute tonsillitis CA09.0,
# CA09.Z, CA03, CA02. | Asthma CA23 | Allergic rhinitis CA08.0 | COPD CA22 | Pleural disease CB2 | Any respiratory CA or CB.
# =====================================================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def q6_yearly() -> pd.DataFrame:
    """Q6.1 Patients per respiratory burden per year, with the all-patient denominator (KPIs, bar chart, year heatmap)"""
    return _run("q6_yearly", f"""
WITH visits AS (
    SELECT
        unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES, is_severe_problem,
        '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
),
flags AS (
    SELECT
        *,
        codes LIKE '%|CA40%'                                                                           AS "Pneumonia",
        codes LIKE ANY ('%|CA07%', '%|CA08.Z%', '%|CA01%', '%|CA04%', '%|CA05%')                        AS "Upper respiratory infection",
        codes LIKE ANY ('%|CA09.0%', '%|CA09.Z%', '%|CA03%', '%|CA02.%')                               AS "Acute tonsillitis",
        codes LIKE '%|CA23%'                                                                           AS "Asthma",
        codes LIKE '%|CA08.0%'                                                                         AS "Allergic rhinitis",
        codes LIKE '%|CA22%'                                                                           AS "COPD",
        codes LIKE '%|CB2%'                                                                            AS "Pleural disease",
        codes LIKE ANY ('%|CA%', '%|CB%')                                                              AS "Any respiratory",
        "Any respiratory" AND NOT ("Pneumonia" OR "Upper respiratory infection" OR "Acute tonsillitis" OR "Asthma" OR "Allergic rhinitis" OR "COPD" OR "Pleural disease") AS "Other respiratory"
    FROM visits
),
dx AS (
    SELECT *
    FROM flags
    UNPIVOT (is_match FOR condition_name IN ("Pneumonia", "Upper respiratory infection", "Acute tonsillitis", "Asthma", "Allergic rhinitis", "COPD", "Pleural disease", "Other respiratory", "Any respiratory"))
    WHERE is_match
),
all_pt AS (
    SELECT
        COALESCE(TO_CHAR(visit_year), 'All years') AS period,
        COUNT(DISTINCT unique_patient_id)          AS all_patients
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
      AND ALL_DIAGNOSES IS NOT NULL
    GROUP BY GROUPING SETS ((visit_year), ())
),
resp AS (
    SELECT
        COALESCE(TO_CHAR(visit_year), 'All years') AS period,
        condition_name,
        COUNT(DISTINCT unique_patient_id)          AS patients,
        COUNT(DISTINCT pno)                        AS visits
    FROM dx
    GROUP BY GROUPING SETS ((visit_year, condition_name), (condition_name))
)
SELECT
    r.period,
    r.condition_name,
    r.patients,
    r.visits,
    a.all_patients,
    ROUND(100.0 * r.patients / a.all_patients, 3) AS pct_of_patients
FROM resp r
JOIN all_pt a ON a.period = r.period
ORDER BY r.condition_name, r.period
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q6_age_sex() -> pd.DataFrame:
    """Q6.2 Patients per burden by age group at first diagnosis and sex (all years), with origin"""
    return _run("q6_age_sex", f"""
WITH visits AS (
    SELECT
        unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES, is_severe_problem,
        '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
      AND age_in_years_float IS NOT NULL
),
flags AS (
    SELECT
        *,
        codes LIKE '%|CA40%'                                                                           AS "Pneumonia",
        codes LIKE ANY ('%|CA07%', '%|CA08.Z%', '%|CA01%', '%|CA04%', '%|CA05%')                        AS "Upper respiratory infection",
        codes LIKE ANY ('%|CA09.0%', '%|CA09.Z%', '%|CA03%', '%|CA02.%')                               AS "Acute tonsillitis",
        codes LIKE '%|CA23%'                                                                           AS "Asthma",
        codes LIKE '%|CA08.0%'                                                                         AS "Allergic rhinitis",
        codes LIKE '%|CA22%'                                                                           AS "COPD",
        codes LIKE '%|CB2%'                                                                            AS "Pleural disease",
        codes LIKE ANY ('%|CA%', '%|CB%')                                                              AS "Any respiratory",
        "Any respiratory" AND NOT ("Pneumonia" OR "Upper respiratory infection" OR "Acute tonsillitis" OR "Asthma" OR "Allergic rhinitis" OR "COPD" OR "Pleural disease") AS "Other respiratory"
    FROM visits
),
dx AS (
    SELECT *
    FROM flags
    UNPIVOT (is_match FOR condition_name IN ("Pneumonia", "Upper respiratory infection", "Acute tonsillitis", "Asthma", "Allergic rhinitis", "COPD", "Pleural disease", "Other respiratory", "Any respiratory"))
    WHERE is_match
),
first_dx AS (
    SELECT
        condition_name,
        unique_patient_id,
        sex,
        county,
        age_in_years_float,
        ROW_NUMBER() OVER (PARTITION BY condition_name, unique_patient_id ORDER BY visit_year, age_in_years_float) AS rn
    FROM dx
)
SELECT
    condition_name,
    CASE
        WHEN age_in_years_float < 5  THEN 'Toddler (0-4)'
        WHEN age_in_years_float < 13 THEN 'Child (5-12)'
        WHEN age_in_years_float < 18 THEN 'Adolescent (13-17)'
        WHEN age_in_years_float < 25 THEN 'Youth (18-24)'
        WHEN age_in_years_float < 35 THEN 'Young Adult (25-34)'
        WHEN age_in_years_float < 45 THEN 'Adult (35-44)'
        WHEN age_in_years_float < 55 THEN 'Middle Age (45-54)'
        ELSE '55+'
    END AS age_group_at_first_diagnosis,
    sex,
    COUNT(DISTINCT unique_patient_id)                  AS patients,
    COUNT_IF(county = 'Nandi')                         AS from_nandi,
    COUNT_IF(county <> 'Nandi' AND county IS NOT NULL) AS from_outside_nandi
FROM first_dx
WHERE rn = 1
GROUP BY condition_name, age_group_at_first_diagnosis, sex
ORDER BY condition_name, MIN(age_in_years_float), sex
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q6_wards() -> pd.DataFrame:
    """Q6.3 Patients per ward and burden (Nandi residents, 2024-2025), with ward geolocation for the map"""
    return _run("q6_wards", f"""
WITH visits AS (
    SELECT
        unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES, is_severe_problem,
        '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
      AND county = 'Nandi'
      AND ward IS NOT NULL
      AND visit_year IN (2024, 2025)
),
flags AS (
    SELECT
        *,
        codes LIKE '%|CA40%'                                                                           AS "Pneumonia",
        codes LIKE ANY ('%|CA07%', '%|CA08.Z%', '%|CA01%', '%|CA04%', '%|CA05%')                        AS "Upper respiratory infection",
        codes LIKE ANY ('%|CA09.0%', '%|CA09.Z%', '%|CA03%', '%|CA02.%')                               AS "Acute tonsillitis",
        codes LIKE '%|CA23%'                                                                           AS "Asthma",
        codes LIKE '%|CA08.0%'                                                                         AS "Allergic rhinitis",
        codes LIKE '%|CA22%'                                                                           AS "COPD",
        codes LIKE '%|CB2%'                                                                            AS "Pleural disease",
        codes LIKE ANY ('%|CA%', '%|CB%')                                                              AS "Any respiratory",
        "Any respiratory" AND NOT ("Pneumonia" OR "Upper respiratory infection" OR "Acute tonsillitis" OR "Asthma" OR "Allergic rhinitis" OR "COPD" OR "Pleural disease") AS "Other respiratory"
    FROM visits
),
dx AS (
    SELECT *
    FROM flags
    UNPIVOT (is_match FOR condition_name IN ("Pneumonia", "Upper respiratory infection", "Acute tonsillitis", "Asthma", "Allergic rhinitis", "COPD", "Pleural disease", "Other respiratory", "Any respiratory"))
    WHERE is_match
),
ward_counts AS (
    SELECT
        ward,
        sub_county,
        condition_name,
        COUNT(DISTINCT unique_patient_id) AS patients
    FROM dx
    GROUP BY ward, sub_county, condition_name
)
SELECT
    c.ward,
    c.sub_county,
    c.condition_name,
    c.patients,
    g.latitude,
    g.longitude,
    g.boundary_geojson
FROM ward_counts c
LEFT JOIN {GEO_TABLE} g
       ON g.geo_level = 'WARD'
      AND g.geocode_status = 'FOUND'
      AND g.county ILIKE 'Nandi'
      AND UPPER(TRIM(g.geo_name))    = UPPER(TRIM(c.ward))
      AND UPPER(TRIM(g.parent_name)) = UPPER(TRIM(c.sub_county))
ORDER BY c.condition_name, c.patients DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q6_returning() -> pd.DataFrame:
    """Q6.4 Who comes back, per burden and age band: seen in both 2024 and 2025, seen more than once"""
    return _run("q6_returning", f"""
WITH visits AS (
    SELECT
        unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES, is_severe_problem,
        '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
      AND age_in_years_float IS NOT NULL
      AND visit_year IN (2024, 2025)
),
flags AS (
    SELECT
        *,
        codes LIKE '%|CA40%'                                                                           AS "Pneumonia",
        codes LIKE ANY ('%|CA07%', '%|CA08.Z%', '%|CA01%', '%|CA04%', '%|CA05%')                        AS "Upper respiratory infection",
        codes LIKE ANY ('%|CA09.0%', '%|CA09.Z%', '%|CA03%', '%|CA02.%')                               AS "Acute tonsillitis",
        codes LIKE '%|CA23%'                                                                           AS "Asthma",
        codes LIKE '%|CA08.0%'                                                                         AS "Allergic rhinitis",
        codes LIKE '%|CA22%'                                                                           AS "COPD",
        codes LIKE '%|CB2%'                                                                            AS "Pleural disease",
        codes LIKE ANY ('%|CA%', '%|CB%')                                                              AS "Any respiratory",
        "Any respiratory" AND NOT ("Pneumonia" OR "Upper respiratory infection" OR "Acute tonsillitis" OR "Asthma" OR "Allergic rhinitis" OR "COPD" OR "Pleural disease") AS "Other respiratory"
    FROM visits
),
dx AS (
    SELECT *
    FROM flags
    UNPIVOT (is_match FOR condition_name IN ("Pneumonia", "Upper respiratory infection", "Acute tonsillitis", "Asthma", "Allergic rhinitis", "COPD", "Pleural disease", "Other respiratory", "Any respiratory"))
    WHERE is_match
),
per_patient AS (
    SELECT
        condition_name,
        unique_patient_id,
        COUNT(DISTINCT visit_year) AS years_seen,
        COUNT(DISTINCT pno)        AS visits,
        CASE
            WHEN MIN(age_in_years_float) < 5  THEN 'Under 5'
            WHEN MIN(age_in_years_float) < 18 THEN '5-17'
            WHEN MIN(age_in_years_float) < 55 THEN '18-54'
            ELSE '55+'
        END AS age_band
    FROM dx
    GROUP BY condition_name, unique_patient_id
)
SELECT
    condition_name,
    age_band,
    COUNT(*)                   AS patients,
    COUNT_IF(years_seen = 2)   AS seen_both_years,
    COUNT_IF(visits > 1)       AS more_than_one_visit,
    SUM(visits)                AS total_visits
FROM per_patient
GROUP BY condition_name, age_band
ORDER BY condition_name, age_band
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q6_severity() -> pd.DataFrame:
    """Q6.5 Pneumonia severity and co-coded HIV by year and age band"""
    return _run("q6_severity", f"""
WITH visits AS (
    SELECT
        unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES, is_severe_problem,
        '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
      AND age_in_years_float IS NOT NULL
),
flags AS (
    SELECT
        *,
        codes LIKE '%|CA40%'                                                                           AS "Pneumonia",
        codes LIKE ANY ('%|CA07%', '%|CA08.Z%', '%|CA01%', '%|CA04%', '%|CA05%')                        AS "Upper respiratory infection",
        codes LIKE ANY ('%|CA09.0%', '%|CA09.Z%', '%|CA03%', '%|CA02.%')                               AS "Acute tonsillitis",
        codes LIKE '%|CA23%'                                                                           AS "Asthma",
        codes LIKE '%|CA08.0%'                                                                         AS "Allergic rhinitis",
        codes LIKE '%|CA22%'                                                                           AS "COPD",
        codes LIKE '%|CB2%'                                                                            AS "Pleural disease",
        codes LIKE ANY ('%|CA%', '%|CB%')                                                              AS "Any respiratory",
        "Any respiratory" AND NOT ("Pneumonia" OR "Upper respiratory infection" OR "Acute tonsillitis" OR "Asthma" OR "Allergic rhinitis" OR "COPD" OR "Pleural disease") AS "Other respiratory"
    FROM visits
)
SELECT
    visit_year,
    CASE WHEN age_in_years_float < 5 THEN 'Under 5' ELSE '5 and over' END                          AS age_band,
    COUNT(DISTINCT unique_patient_id)                                                              AS pneumonia_patients,
    COUNT(DISTINCT IFF(is_severe_problem = TRUE, unique_patient_id, NULL))                         AS severe_patients,
    COUNT(DISTINCT IFF(codes LIKE '%|1C6%', unique_patient_id, NULL))                              AS with_hiv_code,
    COUNT(DISTINCT IFF(ALL_DIAGNOSES ILIKE ANY ('%malnutrition%', '%undernutrition%', '%wasting%'), unique_patient_id, NULL)) AS with_malnutrition_text
FROM flags
WHERE "Pneumonia"
GROUP BY visit_year, age_band
ORDER BY visit_year, age_band
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q6_comorbidity() -> pd.DataFrame:
    """Q6.7 Burden x comorbidity by age band (all years)"""
    return _run("q6_comorbidity", f"""
WITH visits AS (
    SELECT
        unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES, is_severe_problem,
        '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
      AND age_in_years_float IS NOT NULL
),
flags AS (
    SELECT
        *,
        codes LIKE '%|CA40%'                                                                           AS "Pneumonia",
        codes LIKE ANY ('%|CA07%', '%|CA08.Z%', '%|CA01%', '%|CA04%', '%|CA05%')                        AS "Upper respiratory infection",
        codes LIKE ANY ('%|CA09.0%', '%|CA09.Z%', '%|CA03%', '%|CA02.%')                               AS "Acute tonsillitis",
        codes LIKE '%|CA23%'                                                                           AS "Asthma",
        codes LIKE '%|CA08.0%'                                                                         AS "Allergic rhinitis",
        codes LIKE '%|CA22%'                                                                           AS "COPD",
        codes LIKE '%|CB2%'                                                                            AS "Pleural disease",
        codes LIKE ANY ('%|CA%', '%|CB%')                                                              AS "Any respiratory",
        "Any respiratory" AND NOT ("Pneumonia" OR "Upper respiratory infection" OR "Acute tonsillitis" OR "Asthma" OR "Allergic rhinitis" OR "COPD" OR "Pleural disease") AS "Other respiratory"
    FROM visits
),
dx AS (
    SELECT *
    FROM flags
    UNPIVOT (is_match FOR condition_name IN ("Pneumonia", "Upper respiratory infection", "Acute tonsillitis", "Asthma", "Allergic rhinitis", "COPD", "Pleural disease", "Other respiratory", "Any respiratory"))
    WHERE is_match
),
per_patient AS (
    SELECT
        condition_name,
        unique_patient_id,
        CASE
            WHEN MIN(age_in_years_float) < 5  THEN 'Under 5'
            WHEN MIN(age_in_years_float) < 18 THEN '5-17'
            WHEN MIN(age_in_years_float) < 55 THEN '18-54'
            ELSE '55+'
        END AS age_band,
        MAX(IFF(codes LIKE '%|1C6%' OR ALL_DIAGNOSES ILIKE '%hiv%', 1, 0))                                                     AS hiv,
        MAX(IFF(ALL_DIAGNOSES ILIKE '%tubercul%', 1, 0))                                                                       AS tb,
        MAX(IFF(ALL_DIAGNOSES ILIKE ANY ('%malnutrition%', '%undernutrition%', '%wasting%', '%kwashiorkor%', '%marasmus%'), 1, 0)) AS malnutrition,
        MAX(IFF(ALL_DIAGNOSES ILIKE '%anaemia%' OR ALL_DIAGNOSES ILIKE '%anemia%', 1, 0))                                      AS anaemia,
        MAX(IFF(ALL_DIAGNOSES ILIKE '%malaria%', 1, 0))                                                                        AS malaria,
        MAX(IFF(ALL_DIAGNOSES ILIKE '%hypertens%' AND NOT ALL_DIAGNOSES ILIKE ANY ('%pulmonary hypertension%', '%gestational%', '%pregnan%'), 1, 0)) AS hypertension,
        MAX(IFF(ALL_DIAGNOSES ILIKE '%diabetes%' AND NOT ALL_DIAGNOSES ILIKE '%insipidus%', 1, 0))                             AS diabetes,
        MAX(IFF(ALL_DIAGNOSES ILIKE ANY ('%heart failure%', '%cardiac failure%'), 1, 0))                                       AS heart_failure
    FROM dx
    GROUP BY condition_name, unique_patient_id
)
SELECT
    condition_name,
    age_band,
    COUNT(*)           AS patients,
    SUM(hiv)           AS with_hiv,
    SUM(tb)            AS with_tb,
    SUM(malnutrition)  AS with_malnutrition,
    SUM(anaemia)       AS with_anaemia,
    SUM(malaria)       AS with_malaria,
    SUM(hypertension)  AS with_hypertension,
    SUM(diabetes)      AS with_diabetes,
    SUM(heart_failure) AS with_heart_failure
FROM per_patient
GROUP BY condition_name, age_band
ORDER BY condition_name, age_band
    """)



# =====================================================================================================
# TAB 7 · MATERNAL & OBSTETRIC. Source of truth: maternal_queries.sql (Q7.1b, Q7.3, Q7.4, Q7.8, Q7.9, Q7.10b, Q7.16 to Q7.19).
# ICD-11 chapter 18 (JA, JB), counted by patient, default code pair excluded. Categories follow the original Tab 7 map and can overlap.
# Data-quality items (Q7.0c code dictionary, Q7.0d sex/age check, Q7.6, Q7.7, the JA10 "abdominal pregnancy" jump) belong on the Data Quality tab.
# =====================================================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def q7_yearly() -> pd.DataFrame:
    """Q7.1b Patients per maternal category per year, with the all-patient denominator (KPIs, category cards, 2024 vs 2025 chart)"""
    return _run("q7_yearly", f"""
WITH visits AS (
    SELECT
        unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES,
        '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
),
flags AS (
    SELECT
        *,
        codes LIKE ANY ('%|JA0%', '%|JA1%')  AS "Pregnancy loss",
        codes LIKE ANY ('%|JA2%', '%|JA4%', '%|JA60%', '%|JA61%')  AS "Maternal complications",
        codes LIKE ANY ('%|JA62%', '%|JA65%')  AS "Infections in pregnancy",
        codes LIKE ANY ('%|JA63%', '%|JA64%', '%|JB6%')  AS "Maternal medical conditions",
        codes LIKE ANY ('%|JA6B%', '%|JA8%')  AS "High-risk pregnancy",
        codes LIKE ANY ('%|JA66%', '%|JA67%', '%|JB9%')  AS "ANC (Antenatal care)",
        codes LIKE ANY ('%|JB0%', '%|JB2%')  AS "Labour & delivery",
        codes LIKE '%|JB4%'  AS "Postpartum complications",
        codes LIKE ANY ('%|JA%', '%|JB%')  AS "Any maternal",
        "Any maternal" AND NOT ("Pregnancy loss" OR "Maternal complications" OR "Infections in pregnancy" OR "Maternal medical conditions" OR "High-risk pregnancy" OR "ANC (Antenatal care)" OR "Labour & delivery" OR "Postpartum complications")  AS "Other"
    FROM visits
),
dx AS (
    SELECT *
    FROM flags
    UNPIVOT (is_match FOR condition_name IN ("Pregnancy loss", "Maternal complications", "Infections in pregnancy", "Maternal medical conditions", "High-risk pregnancy", "ANC (Antenatal care)", "Labour & delivery", "Postpartum complications", "Other", "Any maternal"))
    WHERE is_match
),
all_pt AS (
    SELECT
        COALESCE(TO_CHAR(visit_year), 'All years') AS period,
        COUNT(DISTINCT unique_patient_id)          AS all_patients
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
      AND ALL_DIAGNOSES IS NOT NULL
    GROUP BY GROUPING SETS ((visit_year), ())
),
mat AS (
    SELECT
        COALESCE(TO_CHAR(visit_year), 'All years') AS period,
        condition_name,
        COUNT(DISTINCT unique_patient_id)          AS patients,
        COUNT(DISTINCT pno)                        AS visits
    FROM dx
    GROUP BY GROUPING SETS ((visit_year, condition_name), (condition_name))
)
SELECT
    m.period,
    m.condition_name,
    m.patients,
    m.visits,
    a.all_patients,
    ROUND(100.0 * m.patients / a.all_patients, 3) AS pct_of_patients
FROM mat m
JOIN all_pt a ON a.period = m.period
ORDER BY m.condition_name, m.period
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q7_conditions() -> pd.DataFrame:
    """Q7.18 Patients per maternal condition per year (category cards)"""
    return _run("q7_conditions", f"""
WITH visits AS (
    SELECT
        unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES,
        '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
),
flags AS (
    SELECT
        *,
        codes LIKE '%|JA05%'  AS "Spontaneous abortion",
        codes LIKE ANY ('%|JA00%', '%|JA01%')  AS "Ectopic & threatened abortion",
        codes LIKE '%|JA10%'  AS "Abdominal/ectopic pregnancy",
        codes LIKE '%|JA2%'  AS "Hypertensive disorder",
        codes LIKE '%|JA4%'  AS "Obstetric haemorrhage",
        codes LIKE '%|JA60%'  AS "Hyperemesis gravidarum",
        codes LIKE '%|JA61%'  AS "Venous complication",
        codes LIKE '%|JA62%'  AS "GU infection (site-specific)",
        codes LIKE '%|JA65%'  AS "GU infection (general)",
        codes LIKE '%|JA63%'  AS "Diabetes in pregnancy",
        codes LIKE '%|JA64%'  AS "Malnutrition in pregnancy",
        codes LIKE '%|JA6B%'  AS "Previous uterine scar",
        codes LIKE ANY ('%|JA66%', '%|JA67%')  AS "Screening & anaesthesia complication",
        codes LIKE '%|JA8%'  AS "Fetal/delivery-problem care",
        codes LIKE '%|JB0%'  AS "Labour/delivery complication",
        codes LIKE '%|JB2%'  AS "Delivery",
        codes LIKE '%|JB40%'  AS "Puerperal infection & sepsis",
        codes LIKE '%|JB41%'  AS "Puerperal mastitis",
        codes LIKE '%|JB6%'  AS "Anaemia & other obstetric condition",
        codes LIKE '%|JB9%'  AS "Antenatal screening finding",
        codes LIKE ANY ('%|JA0%', '%|JA1%') AND NOT (codes LIKE ANY ('%|JA05%', '%|JA00%', '%|JA01%', '%|JA10%'))  AS "Other pregnancy loss",
        codes LIKE '%|JB4%' AND NOT (codes LIKE ANY ('%|JB40%', '%|JB41%'))  AS "Other puerperium complication"
    FROM visits
),
dx AS (
    SELECT *
    FROM flags
    UNPIVOT (is_match FOR condition_name IN ("Spontaneous abortion", "Ectopic & threatened abortion", "Abdominal/ectopic pregnancy", "Hypertensive disorder", "Obstetric haemorrhage", "Hyperemesis gravidarum", "Venous complication", "GU infection (site-specific)", "GU infection (general)", "Diabetes in pregnancy", "Malnutrition in pregnancy", "Previous uterine scar", "Screening & anaesthesia complication", "Fetal/delivery-problem care", "Labour/delivery complication", "Delivery", "Puerperal infection & sepsis", "Puerperal mastitis", "Anaemia & other obstetric condition", "Antenatal screening finding", "Other pregnancy loss", "Other puerperium complication"))
    WHERE is_match
),
mat AS (
    SELECT
        COALESCE(TO_CHAR(visit_year), 'All years') AS period,
        condition_name,
        COUNT(DISTINCT unique_patient_id)          AS patients,
        COUNT(DISTINCT pno)                        AS visits
    FROM dx
    GROUP BY GROUPING SETS ((visit_year, condition_name), (condition_name))
)
SELECT period, condition_name, patients, visits
FROM mat
ORDER BY condition_name, period
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q7_age() -> pd.DataFrame:
    """Q7.4 Maternal patients, visits and age by year"""
    return _run("q7_age", f"""
WITH visits AS (
    SELECT
        unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES,
        '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
),
mat AS (
    SELECT * FROM visits WHERE codes LIKE ANY ('%|JA%', '%|JB%')
),
per_patient AS (
    SELECT
        COALESCE(TO_CHAR(visit_year), 'All years') AS period,
        unique_patient_id,
        MIN(age_in_years_float)                    AS age,
        COUNT(DISTINCT pno)                        AS visits
    FROM mat
    GROUP BY GROUPING SETS ((visit_year, unique_patient_id), (unique_patient_id))
)
SELECT
    period,
    COUNT(*)                         AS maternal_patients,
    SUM(visits)                      AS maternal_visits,
    ROUND(AVG(age), 1)               AS mean_age,
    ROUND(MEDIAN(age), 1)            AS median_age
FROM per_patient
GROUP BY period
ORDER BY period
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q7_loss() -> pd.DataFrame:
    """Q7.8 Pregnancy loss by age band and code"""
    return _run("q7_loss", f"""
WITH visits AS (
    SELECT
        unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES,
        '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
),
flags AS (
    SELECT
        *,
        CASE
            WHEN age_in_years_float < 18 THEN 'Under 18'
            WHEN age_in_years_float < 25 THEN '18-24'
            WHEN age_in_years_float < 35 THEN '25-34'
            WHEN age_in_years_float < 45 THEN '35-44'
            ELSE '45+'
        END                                                              AS age_band,
        codes LIKE '%|JA00%'                                             AS "JA00",
        codes LIKE '%|JA01%'                                             AS "JA01",
        codes LIKE '%|JA02%'                                             AS "JA02",
        codes LIKE '%|JA05%'                                             AS "JA05",
        codes LIKE '%|JA10%'                                             AS "JA10",
        codes LIKE ANY ('%|JA0%', '%|JA1%')
          AND NOT (codes LIKE ANY ('%|JA00%', '%|JA01%', '%|JA02%', '%|JA05%', '%|JA10%')) AS "Other loss code",
        codes LIKE ANY ('%|JA0%', '%|JA1%')                              AS "Any pregnancy loss"
    FROM visits
    WHERE age_in_years_float IS NOT NULL
)
SELECT
    age_band,
    loss_code,
    COUNT(DISTINCT unique_patient_id) AS patients,
    COUNT(DISTINCT pno)               AS visits
FROM flags
UNPIVOT (is_match FOR loss_code IN ("JA00", "JA01", "JA02", "JA05", "JA10", "Other loss code", "Any pregnancy loss"))
WHERE is_match
GROUP BY age_band, loss_code
ORDER BY loss_code, MIN(age_in_years_float)
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q7_htn() -> pd.DataFrame:
    """Q7.19 Hypertensive disorders by type and age band"""
    return _run("q7_htn", f"""
WITH visits AS (
    SELECT
        unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES,
        '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
),
flags AS (
    SELECT
        *,
        CASE
            WHEN age_in_years_float < 18 THEN 'Under 18'
            WHEN age_in_years_float < 25 THEN '18-24'
            WHEN age_in_years_float < 35 THEN '25-34'
            WHEN age_in_years_float < 45 THEN '35-44'
            ELSE '45+'
        END                                                                             AS age_band,
        codes LIKE '%|JA24%'                                                            AS "Pre-eclampsia",
        codes LIKE ANY ('%|JA20.2%', '%|JA20.4%', '%|JA21.0%')                          AS "Pre-existing hypertension / renal",
        (codes || '|') LIKE '%|JA20|%'                                                  AS "Gestational hypertension",
        (codes || '|') LIKE '%|JA21|%'                                                  AS "Superimposed pre-eclampsia",
        codes LIKE '%|JA25%'                                                            AS "Eclampsia",
        codes LIKE '%|JA2%'                                                             AS "Any hypertensive disorder"
    FROM visits
    WHERE sex = 'F'
      AND age_in_years_float IS NOT NULL
)
SELECT
    age_band,
    hypertensive_type,
    COUNT(DISTINCT unique_patient_id) AS patients,
    COUNT(DISTINCT pno)               AS visits
FROM flags
UNPIVOT (is_match FOR hypertensive_type IN ("Pre-eclampsia", "Pre-existing hypertension / renal", "Gestational hypertension", "Superimposed pre-eclampsia", "Eclampsia", "Any hypertensive disorder"))
WHERE is_match
GROUP BY age_band, hypertensive_type
ORDER BY hypertensive_type, MIN(age_in_years_float)
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q7_comorbidity() -> pd.DataFrame:
    """Q7.10b Comorbidity per maternal category"""
    return _run("q7_comorbidity", f"""
WITH visits AS (
    SELECT
        unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES,
        '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
),
per_patient AS (
    SELECT
        unique_patient_id,
        BOOLOR_AGG(codes LIKE ANY ('%|JA0%', '%|JA1%')) AS "Pregnancy loss",
        BOOLOR_AGG(codes LIKE ANY ('%|JA2%', '%|JA4%', '%|JA60%', '%|JA61%')) AS "Maternal complications",
        BOOLOR_AGG(codes LIKE ANY ('%|JA62%', '%|JA65%')) AS "Infections in pregnancy",
        BOOLOR_AGG(codes LIKE ANY ('%|JA63%', '%|JA64%', '%|JB6%')) AS "Maternal medical conditions",
        BOOLOR_AGG(codes LIKE ANY ('%|JA6B%', '%|JA8%')) AS "High-risk pregnancy",
        BOOLOR_AGG(codes LIKE ANY ('%|JA66%', '%|JA67%', '%|JB9%')) AS "ANC (Antenatal care)",
        BOOLOR_AGG(codes LIKE ANY ('%|JB0%', '%|JB2%')) AS "Labour & delivery",
        BOOLOR_AGG(codes LIKE '%|JB4%') AS "Postpartum complications",
        BOOLOR_AGG(codes LIKE ANY ('%|JA%', '%|JB%')) AS "Any maternal",
        BOOLOR_AGG(ALL_DIAGNOSES ILIKE '%diabetes%')                                      AS c_diabetes,
        BOOLOR_AGG(ALL_DIAGNOSES ILIKE '%anaemia%')                                       AS c_anaemia,
        BOOLOR_AGG(ALL_DIAGNOSES ILIKE ANY ('%malaria%', '%plasmodium%'))                 AS c_malaria,
        BOOLOR_AGG(ALL_DIAGNOSES ILIKE '%hiv%')                                           AS c_hiv,
        BOOLOR_AGG(ALL_DIAGNOSES ILIKE '%urinary tract infection%')                       AS c_uti,
        BOOLOR_AGG(ALL_DIAGNOSES ILIKE '%hypertension%')                                  AS c_hypertension,
        BOOLOR_AGG(ALL_DIAGNOSES ILIKE '%sepsis%')                                        AS c_sepsis
    FROM visits
    GROUP BY unique_patient_id
),
flags AS (
    SELECT
        *,
        "Any maternal" AND NOT ("Pregnancy loss" OR "Maternal complications" OR "Infections in pregnancy" OR "Maternal medical conditions" OR "High-risk pregnancy" OR "ANC (Antenatal care)" OR "Labour & delivery" OR "Postpartum complications") AS "Other"
    FROM per_patient
),
dx AS (
    SELECT *
    FROM flags
    UNPIVOT (is_match FOR condition_name IN ("Pregnancy loss", "Maternal complications", "Infections in pregnancy", "Maternal medical conditions", "High-risk pregnancy", "ANC (Antenatal care)", "Labour & delivery", "Postpartum complications", "Other", "Any maternal"))
    WHERE is_match
)
SELECT
    condition_name,
    COUNT(*)                       AS patients,
    COUNT_IF(c_diabetes)           AS with_diabetes,
    COUNT_IF(c_anaemia)            AS with_anaemia,
    COUNT_IF(c_malaria)            AS with_malaria,
    COUNT_IF(c_hiv)                AS with_hiv,
    COUNT_IF(c_uti)                AS with_uti,
    COUNT_IF(c_hypertension)       AS with_hypertension,
    COUNT_IF(c_sepsis)             AS with_sepsis
FROM dx
GROUP BY condition_name
ORDER BY patients DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q7_teen() -> pd.DataFrame:
    """Q7.3 Teen maternal patients by year and age group"""
    return _run("q7_teen", f"""
WITH visits AS (
    SELECT
        unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES,
        '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
)
SELECT
    visit_year,
    CASE
        WHEN age_in_years_float < 16 THEN '13-15'
        WHEN age_in_years_float < 18 THEN '16-17'
        ELSE '18-19'
    END                                  AS age_group,
    COUNT(DISTINCT unique_patient_id)    AS patients,
    COUNT(DISTINCT pno)                  AS visits
FROM visits
WHERE codes LIKE ANY ('%|JA%', '%|JB%')
  AND age_in_years_float BETWEEN 13 AND 19.999
GROUP BY visit_year, age_group
ORDER BY visit_year, age_group
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q7_teen_share() -> pd.DataFrame:
    """Q7.9 Teen maternal share of teen OPD patients"""
    return _run("q7_teen_share", f"""
WITH visits AS (
    SELECT
        unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES,
        '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
),
teen AS (
    SELECT
        visit_year,
        COUNT(DISTINCT unique_patient_id)                                                                  AS teen_opd_patients,
        COUNT(DISTINCT pno)                                                                                AS teen_opd_visits,
        COUNT(DISTINCT IFF(codes LIKE ANY ('%|JA%', '%|JB%'), unique_patient_id, NULL))                    AS teen_maternal_patients,
        COUNT(DISTINCT IFF(codes LIKE ANY ('%|JA%', '%|JB%'), pno, NULL))                                  AS teen_maternal_visits
    FROM visits
    WHERE age_in_years_float BETWEEN 13 AND 19.999
    GROUP BY visit_year
)
SELECT
    visit_year,
    teen_maternal_patients,
    teen_opd_patients,
    ROUND(100.0 * teen_maternal_patients / NULLIF(teen_opd_patients, 0), 2) AS pct_of_teen_patients,
    teen_maternal_visits,
    teen_opd_visits,
    ROUND(100.0 * teen_maternal_visits / NULLIF(teen_opd_visits, 0), 2)     AS pct_of_teen_visits
FROM teen
ORDER BY visit_year
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q7_wards() -> pd.DataFrame:
    """Q7.16 Maternal patients per ward and category, with ward geolocation for the map"""
    return _run("q7_wards", f"""
WITH visits AS (
    SELECT
        unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES,
        '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
      AND county = 'Nandi'
      AND ward IS NOT NULL
      AND visit_year IN (2024, 2025)
),
flags AS (
    SELECT
        *,
        codes LIKE ANY ('%|JA0%', '%|JA1%')  AS "Pregnancy loss",
        codes LIKE ANY ('%|JA2%', '%|JA4%', '%|JA60%', '%|JA61%')  AS "Maternal complications",
        codes LIKE ANY ('%|JA62%', '%|JA65%')  AS "Infections in pregnancy",
        codes LIKE ANY ('%|JA63%', '%|JA64%', '%|JB6%')  AS "Maternal medical conditions",
        codes LIKE ANY ('%|JA6B%', '%|JA8%')  AS "High-risk pregnancy",
        codes LIKE ANY ('%|JA66%', '%|JA67%', '%|JB9%')  AS "ANC (Antenatal care)",
        codes LIKE ANY ('%|JB0%', '%|JB2%')  AS "Labour & delivery",
        codes LIKE '%|JB4%'  AS "Postpartum complications",
        codes LIKE ANY ('%|JA%', '%|JB%')  AS "Any maternal",
        "Any maternal" AND NOT ("Pregnancy loss" OR "Maternal complications" OR "Infections in pregnancy" OR "Maternal medical conditions" OR "High-risk pregnancy" OR "ANC (Antenatal care)" OR "Labour & delivery" OR "Postpartum complications")  AS "Other"
    FROM visits
),
dx AS (
    SELECT *
    FROM flags
    UNPIVOT (is_match FOR condition_name IN ("Pregnancy loss", "Maternal complications", "Infections in pregnancy", "Maternal medical conditions", "High-risk pregnancy", "ANC (Antenatal care)", "Labour & delivery", "Postpartum complications", "Other", "Any maternal"))
    WHERE is_match
),
ward_counts AS (
    SELECT
        ward,
        sub_county,
        condition_name,
        COUNT(DISTINCT unique_patient_id) AS patients
    FROM dx
    GROUP BY ward, sub_county, condition_name
)
SELECT
    c.ward,
    c.sub_county,
    c.condition_name,
    c.patients,
    g.latitude,
    g.longitude,
    g.boundary_geojson
FROM ward_counts c
LEFT JOIN {GEO_TABLE} g
       ON g.geo_level = 'WARD'
      AND g.geocode_status = 'FOUND'
      AND g.county ILIKE 'Nandi'
      AND UPPER(TRIM(g.geo_name))    = UPPER(TRIM(c.ward))
      AND UPPER(TRIM(g.parent_name)) = UPPER(TRIM(c.sub_county))
ORDER BY c.condition_name, c.patients DESC
    """)


@st.cache_data(ttl=3600, show_spinner=False)
def q7_returning() -> pd.DataFrame:
    """Q7.17 Who comes back, per category and age band"""
    return _run("q7_returning", f"""
WITH visits AS (
    SELECT
        unique_patient_id, pno, visit_year, sex, age_in_years_float, county, sub_county, ward, ALL_DIAGNOSES,
        '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
      AND age_in_years_float IS NOT NULL
      AND visit_year IN (2024, 2025)
),
flags AS (
    SELECT
        *,
        codes LIKE ANY ('%|JA0%', '%|JA1%')  AS "Pregnancy loss",
        codes LIKE ANY ('%|JA2%', '%|JA4%', '%|JA60%', '%|JA61%')  AS "Maternal complications",
        codes LIKE ANY ('%|JA62%', '%|JA65%')  AS "Infections in pregnancy",
        codes LIKE ANY ('%|JA63%', '%|JA64%', '%|JB6%')  AS "Maternal medical conditions",
        codes LIKE ANY ('%|JA6B%', '%|JA8%')  AS "High-risk pregnancy",
        codes LIKE ANY ('%|JA66%', '%|JA67%', '%|JB9%')  AS "ANC (Antenatal care)",
        codes LIKE ANY ('%|JB0%', '%|JB2%')  AS "Labour & delivery",
        codes LIKE '%|JB4%'  AS "Postpartum complications",
        codes LIKE ANY ('%|JA%', '%|JB%')  AS "Any maternal",
        "Any maternal" AND NOT ("Pregnancy loss" OR "Maternal complications" OR "Infections in pregnancy" OR "Maternal medical conditions" OR "High-risk pregnancy" OR "ANC (Antenatal care)" OR "Labour & delivery" OR "Postpartum complications")  AS "Other"
    FROM visits
),
dx AS (
    SELECT *
    FROM flags
    UNPIVOT (is_match FOR condition_name IN ("Pregnancy loss", "Maternal complications", "Infections in pregnancy", "Maternal medical conditions", "High-risk pregnancy", "ANC (Antenatal care)", "Labour & delivery", "Postpartum complications", "Other", "Any maternal"))
    WHERE is_match
),
per_patient AS (
    SELECT
        condition_name,
        unique_patient_id,
        COUNT(DISTINCT visit_year) AS years_seen,
        COUNT(DISTINCT pno)        AS visits,
        CASE
            WHEN MIN(age_in_years_float) < 18 THEN 'Under 18'
            WHEN MIN(age_in_years_float) < 25 THEN '18-24'
            WHEN MIN(age_in_years_float) < 35 THEN '25-34'
            ELSE '35+'
        END AS age_band
    FROM dx
    GROUP BY condition_name, unique_patient_id
)
SELECT
    condition_name,
    age_band,
    COUNT(*)                   AS patients,
    COUNT_IF(years_seen = 2)   AS seen_both_years,
    COUNT_IF(visits > 1)       AS more_than_one_visit,
    SUM(visits)                AS total_visits
FROM per_patient
GROUP BY condition_name, age_band
ORDER BY condition_name, age_band
    """)



@st.cache_data(ttl=3600, show_spinner=False)
def q7_diabetes() -> pd.DataFrame:
    """Q7.11 Diabetes type among maternal patients (original query, unchanged: text-based, patient-distinct)"""
    return _run("q7_diabetes", f"""
WITH maternal_patients AS (
    SELECT DISTINCT o.unique_patient_id
    FROM {TABLE} o,
         LATERAL FLATTEN(INPUT => ARRAY_CONSTRUCT(
             o.diagnosis_1_icd11, o.diagnosis_2_icd11, o.diagnosis_3_icd11, o.diagnosis_4_icd11,
             o.diagnosis_5_icd11, o.diagnosis_6_icd11, o.diagnosis_7_icd11, o.diagnosis_8_icd11
         )) f
    WHERE LEFT(f.value::string, 2) IN ('JA', 'JB')
)
SELECT
    CASE
        WHEN o.ALL_DIAGNOSES ILIKE '%gestational diabetes%'  THEN 'Gestational diabetes'
        WHEN o.ALL_DIAGNOSES ILIKE '%type 1 diabetes%'        THEN 'Pre-existing Type 1'
        WHEN o.ALL_DIAGNOSES ILIKE '%type 2 diabetes%'        THEN 'Pre-existing Type 2'
        ELSE 'Diabetes, unspecified type'
    END                                        AS diabetes_type,
    COUNT(DISTINCT p.unique_patient_id)        AS patients
FROM maternal_patients p
JOIN {TABLE} o ON p.unique_patient_id = o.unique_patient_id
WHERE o.ALL_DIAGNOSES ILIKE '%diabetes%'
GROUP BY diabetes_type
ORDER BY patients DESC
    """)



# =================================================================== TAB 8: INJURY & TRAUMA
# Injury and trauma, counted by patient, default code pair excluded for the general injury queries (SGBV queries do not apply it, so they
# reconcile to the original 110 patients). Diagnosis text is matched one diagnosis slot at a time. Source: injury_queries.sql.
# Q8.2, Q8.3 (internal SGBV detail) and Q8.0, Q8.16 (dictionaries for the Data Quality tab) are deliberately not run here.

@st.cache_data(ttl=3600, show_spinner=False)
def q8_yearly() -> pd.DataFrame:
    """Q8.1 Injury patients by type and year"""
    return _run("q8_yearly", f"""
WITH src AS (
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem,
           ALL_DIAGNOSES AS all_dx,
           diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4, diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
),
long AS (
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_1 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_2 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_3 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_4 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_5 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_6 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_7 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_8 AS dx FROM src
),
flags AS (
    SELECT
        *,
        (dx ILIKE '%fracture%' AND NOT (dx ILIKE ANY ('%union of fracture%', '%nontraumatic%', '%birth injury%'))) AS is_fracture,
        (dx ILIKE '%wound%' AND NOT (dx ILIKE ANY ('%woundwort%', '%obstetric%', '%surgical wound%')))            AS is_open_wound,
        (dx ILIKE '%burn%' AND NOT (dx ILIKE ANY ('%heartburn%', '%burning%')))                                    AS is_burn,
        (dx ILIKE ANY ('%land transport traffic%', '%land transport crash%', '%road traffic%', '%motor vehicle%', '%motorcycle%', '%pedestrian%', '%person on foot injured in transport%')
         AND NOT (dx ILIKE ANY ('%assault%', '%self-harm%', '%examination%', '%observation%')))                  AS is_rta,
        (dx ILIKE ANY ('%dislocation%', '%sprain%') AND NOT (dx ILIKE ANY ('%congenital%', '%birth injury%', '%of lens%'))) AS is_dislocation_sprain,
        (is_fracture OR is_open_wound OR is_burn OR is_rta OR is_dislocation_sprain)                                        AS is_any_injury
    FROM long
)
SELECT
    visit_year,
    COUNT(DISTINCT unique_patient_id)        AS all_patients,
    COUNT(DISTINCT IFF(is_any_injury, unique_patient_id, NULL))       AS any_injury,
    COUNT(DISTINCT IFF(is_fracture, unique_patient_id, NULL))        AS fractures,
    COUNT(DISTINCT IFF(is_open_wound, unique_patient_id, NULL))      AS open_wounds,
    COUNT(DISTINCT IFF(is_burn, unique_patient_id, NULL))            AS burns,
    COUNT(DISTINCT IFF(is_rta, unique_patient_id, NULL))              AS rta,
    COUNT(DISTINCT IFF(is_dislocation_sprain, unique_patient_id, NULL)) AS dislocation_sprain
FROM flags
GROUP BY visit_year
ORDER BY visit_year;
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q8_top() -> pd.DataFrame:
    """Q8.13 Top injury conditions by patient"""
    return _run("q8_top", f"""
WITH base AS (
    SELECT
        unique_patient_id, sex, age_in_years_float,
        ARRAY_CONSTRUCT(diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4,
                        diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8) AS dxs
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
),
long AS (
    SELECT b.unique_patient_id, b.sex, b.age_in_years_float, f.value::string AS condition
    FROM base b, LATERAL FLATTEN(INPUT => b.dxs) f
    WHERE f.value IS NOT NULL
      AND f.value::string ILIKE ANY ('%fracture%', '%wound%', '%burn%', '%dislocation%', '%sprain%',
                                     '%strain%', '%contusion%', '%laceration%', '%injur%')
      AND f.value::string NOT ILIKE '%woundwort%'
      AND f.value::string NOT ILIKE '%heartburn%'
      AND f.value::string NOT ILIKE '%burning%'
      AND f.value::string NOT ILIKE '%stress%'
      AND NOT (f.value::string ILIKE ANY ('%epilepsy%', '%prenatal%', '%perinatal%'))
),
patient_condition AS (
    SELECT condition, unique_patient_id, MAX(sex) AS sex, MIN(age_in_years_float) AS age
    FROM long
    GROUP BY condition, unique_patient_id
)
SELECT
    condition,
    COUNT(*)                   AS patients,
    COUNT_IF(sex = 'F')        AS female,
    COUNT_IF(sex = 'M')        AS male,
    ROUND(AVG(age), 1)         AS mean_age
FROM patient_condition
GROUP BY condition
ORDER BY patients DESC
LIMIT 20;
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q8_age() -> pd.DataFrame:
    """Q8.14 Injury type by age band"""
    return _run("q8_age", f"""
WITH src AS (
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem,
           ALL_DIAGNOSES AS all_dx,
           diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4, diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
),
long AS (
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_1 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_2 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_3 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_4 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_5 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_6 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_7 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_8 AS dx FROM src
),
flags AS (
    SELECT
        *,
        (dx ILIKE '%fracture%' AND NOT (dx ILIKE ANY ('%union of fracture%', '%nontraumatic%', '%birth injury%'))) AS is_fracture,
        (dx ILIKE '%wound%' AND NOT (dx ILIKE ANY ('%woundwort%', '%obstetric%', '%surgical wound%')))            AS is_open_wound,
        (dx ILIKE '%burn%' AND NOT (dx ILIKE ANY ('%heartburn%', '%burning%')))                                    AS is_burn,
        (dx ILIKE ANY ('%land transport traffic%', '%land transport crash%', '%road traffic%', '%motor vehicle%', '%motorcycle%', '%pedestrian%', '%person on foot injured in transport%')
         AND NOT (dx ILIKE ANY ('%assault%', '%self-harm%', '%examination%', '%observation%')))                  AS is_rta,
        (dx ILIKE ANY ('%dislocation%', '%sprain%') AND NOT (dx ILIKE ANY ('%congenital%', '%birth injury%', '%of lens%'))) AS is_dislocation_sprain,
        (is_fracture OR is_open_wound OR is_burn OR is_rta OR is_dislocation_sprain)                                        AS is_any_injury
    FROM long
),
pat AS (
    SELECT
        unique_patient_id,
        MIN(age_in_years_float)        AS age,
        BOOLOR_AGG(is_fracture)          AS "Fractures",
        BOOLOR_AGG(is_open_wound)        AS "Open wounds",
        BOOLOR_AGG(is_burn)              AS "Burns",
        BOOLOR_AGG(is_rta)                AS "RTA",
        BOOLOR_AGG(is_dislocation_sprain) AS "Dislocation / sprain"
    FROM flags
    WHERE age_in_years_float IS NOT NULL
    GROUP BY unique_patient_id
),
banded AS (
    SELECT
        *,
        CASE
            WHEN age < 5  THEN 'Under 5'
            WHEN age < 13 THEN '5-12'
            WHEN age < 18 THEN '13-17'
            WHEN age < 25 THEN '18-24'
            WHEN age < 35 THEN '25-34'
            WHEN age < 45 THEN '35-44'
            WHEN age < 55 THEN '45-54'
            ELSE '55+'
        END AS age_band
    FROM pat
),
dx_long AS (
    SELECT *
    FROM banded
    UNPIVOT (is_match FOR injury_type IN ("Fractures", "Open wounds", "Burns", "RTA", "Dislocation / sprain"))
    WHERE is_match
)
SELECT
    injury_type,
    age_band,
    COUNT(DISTINCT unique_patient_id) AS patients
FROM dx_long
GROUP BY injury_type, age_band
ORDER BY injury_type, MIN(age);
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q8_severity() -> pd.DataFrame:
    """Q8.12 Share of fracture, burn and RTA patients flagged severe"""
    return _run("q8_severity", f"""
WITH src AS (
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem,
           ALL_DIAGNOSES AS all_dx,
           diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4, diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
),
long AS (
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_1 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_2 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_3 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_4 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_5 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_6 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_7 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_8 AS dx FROM src
),
flags AS (
    SELECT
        *,
        (dx ILIKE '%fracture%' AND NOT (dx ILIKE ANY ('%union of fracture%', '%nontraumatic%', '%birth injury%'))) AS is_fracture,
        (dx ILIKE '%wound%' AND NOT (dx ILIKE ANY ('%woundwort%', '%obstetric%', '%surgical wound%')))            AS is_open_wound,
        (dx ILIKE '%burn%' AND NOT (dx ILIKE ANY ('%heartburn%', '%burning%')))                                    AS is_burn,
        (dx ILIKE ANY ('%land transport traffic%', '%land transport crash%', '%road traffic%', '%motor vehicle%', '%motorcycle%', '%pedestrian%', '%person on foot injured in transport%')
         AND NOT (dx ILIKE ANY ('%assault%', '%self-harm%', '%examination%', '%observation%')))                  AS is_rta,
        (dx ILIKE ANY ('%dislocation%', '%sprain%') AND NOT (dx ILIKE ANY ('%congenital%', '%birth injury%', '%of lens%'))) AS is_dislocation_sprain,
        (is_fracture OR is_open_wound OR is_burn OR is_rta OR is_dislocation_sprain)                                        AS is_any_injury
    FROM long
)
SELECT 'Fractures' AS condition, visit_year,
    COUNT(DISTINCT unique_patient_id)                                      AS patients,
    COUNT(DISTINCT IFF(is_severe_problem = TRUE, unique_patient_id, NULL)) AS severe_patients
FROM flags WHERE is_fracture GROUP BY visit_year
UNION ALL
SELECT 'Burns', visit_year, COUNT(DISTINCT unique_patient_id),
    COUNT(DISTINCT IFF(is_severe_problem = TRUE, unique_patient_id, NULL))
FROM flags WHERE is_burn GROUP BY visit_year
UNION ALL
SELECT 'RTA', visit_year, COUNT(DISTINCT unique_patient_id),
    COUNT(DISTINCT IFF(is_severe_problem = TRUE, unique_patient_id, NULL))
FROM flags WHERE is_rta GROUP BY visit_year
ORDER BY condition, visit_year;
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q8_sepsis() -> pd.DataFrame:
    """Q8.4 Sepsis among open-wound patients"""
    return _run("q8_sepsis", f"""
WITH src AS (
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem,
           ALL_DIAGNOSES AS all_dx,
           diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4, diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
),
long AS (
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_1 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_2 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_3 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_4 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_5 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_6 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_7 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_8 AS dx FROM src
),
flags AS (
    SELECT
        *,
        (dx ILIKE '%fracture%' AND NOT (dx ILIKE ANY ('%union of fracture%', '%nontraumatic%', '%birth injury%'))) AS is_fracture,
        (dx ILIKE '%wound%' AND NOT (dx ILIKE ANY ('%woundwort%', '%obstetric%', '%surgical wound%')))            AS is_open_wound,
        (dx ILIKE '%burn%' AND NOT (dx ILIKE ANY ('%heartburn%', '%burning%')))                                    AS is_burn,
        (dx ILIKE ANY ('%land transport traffic%', '%land transport crash%', '%road traffic%', '%motor vehicle%', '%motorcycle%', '%pedestrian%', '%person on foot injured in transport%')
         AND NOT (dx ILIKE ANY ('%assault%', '%self-harm%', '%examination%', '%observation%')))                  AS is_rta,
        (dx ILIKE ANY ('%dislocation%', '%sprain%') AND NOT (dx ILIKE ANY ('%congenital%', '%birth injury%', '%of lens%'))) AS is_dislocation_sprain,
        (is_fracture OR is_open_wound OR is_burn OR is_rta OR is_dislocation_sprain)                                        AS is_any_injury
    FROM long
)
SELECT
    visit_year,
    COUNT(DISTINCT IFF(is_open_wound, unique_patient_id, NULL))                                                   AS open_wound_patients,
    COUNT(DISTINCT IFF(is_open_wound AND all_dx ILIKE '%sepsis%', unique_patient_id, NULL))                       AS with_sepsis,
    ROUND(COUNT(DISTINCT IFF(is_open_wound AND all_dx ILIKE '%sepsis%', unique_patient_id, NULL))
          / NULLIF(COUNT(DISTINCT IFF(is_open_wound, unique_patient_id, NULL)), 0) * 100, 1)                      AS pct_with_sepsis
FROM flags
GROUP BY visit_year
ORDER BY visit_year;
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q8_origin() -> pd.DataFrame:
    """Q8.11 Patients by county of residence"""
    return _run("q8_origin", f"""
WITH src AS (
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem,
           ALL_DIAGNOSES AS all_dx,
           diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4, diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
),
long AS (
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_1 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_2 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_3 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_4 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_5 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_6 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_7 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_8 AS dx FROM src
),
flags AS (
    SELECT
        *,
        (dx ILIKE '%fracture%' AND NOT (dx ILIKE ANY ('%union of fracture%', '%nontraumatic%', '%birth injury%'))) AS is_fracture,
        (dx ILIKE '%wound%' AND NOT (dx ILIKE ANY ('%woundwort%', '%obstetric%', '%surgical wound%')))            AS is_open_wound,
        (dx ILIKE '%burn%' AND NOT (dx ILIKE ANY ('%heartburn%', '%burning%')))                                    AS is_burn,
        (dx ILIKE ANY ('%land transport traffic%', '%land transport crash%', '%road traffic%', '%motor vehicle%', '%motorcycle%', '%pedestrian%', '%person on foot injured in transport%')
         AND NOT (dx ILIKE ANY ('%assault%', '%self-harm%', '%examination%', '%observation%')))                  AS is_rta,
        (dx ILIKE ANY ('%dislocation%', '%sprain%') AND NOT (dx ILIKE ANY ('%congenital%', '%birth injury%', '%of lens%'))) AS is_dislocation_sprain,
        (is_fracture OR is_open_wound OR is_burn OR is_rta OR is_dislocation_sprain)                                        AS is_any_injury
    FROM long
)
SELECT
    CASE WHEN county IS NULL THEN 'Unknown'
         WHEN county = 'Nandi' THEN 'From Nandi'
         ELSE 'Outside Nandi' END   AS origin,
    COUNT(DISTINCT IFF(is_fracture, unique_patient_id, NULL))   AS fractures,
    COUNT(DISTINCT IFF(is_open_wound, unique_patient_id, NULL)) AS open_wounds,
    COUNT(DISTINCT IFF(is_burn, unique_patient_id, NULL))       AS burns
FROM flags
GROUP BY origin
ORDER BY origin;
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q8_wards() -> pd.DataFrame:
    """Q8.15 Injury patients per ward and type, with ward geolocation for the map"""
    return _run("q8_wards", f"""
WITH src AS (
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem,
           ALL_DIAGNOSES AS all_dx,
           diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4, diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
      AND county = 'Nandi'
      AND ward IS NOT NULL
      AND visit_year IN (2024, 2025)
),
long AS (
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_1 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_2 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_3 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_4 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_5 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_6 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_7 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_8 AS dx FROM src
),
flags AS (
    SELECT
        *,
        (dx ILIKE '%fracture%' AND NOT (dx ILIKE ANY ('%union of fracture%', '%nontraumatic%', '%birth injury%'))) AS is_fracture,
        (dx ILIKE '%wound%' AND NOT (dx ILIKE ANY ('%woundwort%', '%obstetric%', '%surgical wound%')))            AS is_open_wound,
        (dx ILIKE '%burn%' AND NOT (dx ILIKE ANY ('%heartburn%', '%burning%')))                                    AS is_burn,
        (dx ILIKE ANY ('%land transport traffic%', '%land transport crash%', '%road traffic%', '%motor vehicle%', '%motorcycle%', '%pedestrian%', '%person on foot injured in transport%')
         AND NOT (dx ILIKE ANY ('%assault%', '%self-harm%', '%examination%', '%observation%')))                  AS is_rta,
        (dx ILIKE ANY ('%dislocation%', '%sprain%') AND NOT (dx ILIKE ANY ('%congenital%', '%birth injury%', '%of lens%'))) AS is_dislocation_sprain,
        (is_fracture OR is_open_wound OR is_burn OR is_rta OR is_dislocation_sprain)                                        AS is_any_injury
    FROM long
),
ward_counts AS (
    SELECT
        ward, sub_county,
        COUNT(DISTINCT IFF(is_any_injury, unique_patient_id, NULL))          AS "Any injury",
        COUNT(DISTINCT IFF(is_fracture, unique_patient_id, NULL))           AS "Fractures",
        COUNT(DISTINCT IFF(is_open_wound, unique_patient_id, NULL))         AS "Open wounds",
        COUNT(DISTINCT IFF(is_burn, unique_patient_id, NULL))               AS "Burns",
        COUNT(DISTINCT IFF(is_rta, unique_patient_id, NULL))                 AS "RTA",
        COUNT(DISTINCT IFF(is_dislocation_sprain, unique_patient_id, NULL))  AS "Dislocation / sprain"
    FROM flags
    GROUP BY ward, sub_county
),
long_counts AS (
    SELECT *
    FROM ward_counts
    UNPIVOT (patients FOR condition_name IN ("Any injury", "Fractures", "Open wounds", "Burns", "RTA", "Dislocation / sprain"))
    WHERE patients > 0
)
SELECT
    c.ward,
    c.sub_county,
    c.condition_name,
    c.patients,
    g.latitude,
    g.longitude,
    g.boundary_geojson
FROM long_counts c
LEFT JOIN {GEO_TABLE} g
       ON g.geo_level = 'WARD'
      AND g.geocode_status = 'FOUND'
      AND g.county ILIKE 'Nandi'
      AND UPPER(TRIM(g.geo_name))    = UPPER(TRIM(c.ward))
      AND UPPER(TRIM(g.parent_name)) = UPPER(TRIM(c.sub_county))
ORDER BY c.condition_name, c.patients DESC;
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q8_burns() -> pd.DataFrame:
    """Q8.10 Burn patients by age and sex"""
    return _run("q8_burns", f"""
WITH src AS (
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem,
           ALL_DIAGNOSES AS all_dx,
           diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4, diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
),
long AS (
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_1 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_2 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_3 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_4 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_5 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_6 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_7 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, sex, age_in_years_float, county, sub_county, ward, is_severe_problem, all_dx, diagnosis_8 AS dx FROM src
),
flags AS (
    SELECT
        *,
        (dx ILIKE '%fracture%' AND NOT (dx ILIKE ANY ('%union of fracture%', '%nontraumatic%', '%birth injury%'))) AS is_fracture,
        (dx ILIKE '%wound%' AND NOT (dx ILIKE ANY ('%woundwort%', '%obstetric%', '%surgical wound%')))            AS is_open_wound,
        (dx ILIKE '%burn%' AND NOT (dx ILIKE ANY ('%heartburn%', '%burning%')))                                    AS is_burn,
        (dx ILIKE ANY ('%land transport traffic%', '%land transport crash%', '%road traffic%', '%motor vehicle%', '%motorcycle%', '%pedestrian%', '%person on foot injured in transport%')
         AND NOT (dx ILIKE ANY ('%assault%', '%self-harm%', '%examination%', '%observation%')))                  AS is_rta,
        (dx ILIKE ANY ('%dislocation%', '%sprain%') AND NOT (dx ILIKE ANY ('%congenital%', '%birth injury%', '%of lens%'))) AS is_dislocation_sprain,
        (is_fracture OR is_open_wound OR is_burn OR is_rta OR is_dislocation_sprain)                                        AS is_any_injury
    FROM long
),
burn_patients AS (
    SELECT unique_patient_id, MIN(age_in_years_float) AS age, MAX(sex) AS sex
    FROM flags
    WHERE is_burn AND age_in_years_float IS NOT NULL
    GROUP BY unique_patient_id
)
SELECT
    CASE
        WHEN age < 5  THEN 'Toddler (0-4)'
        WHEN age < 13 THEN 'Child (5-12)'
        WHEN age < 18 THEN 'Adolescent (13-17)'
        WHEN age < 25 THEN 'Youth (18-24)'
        WHEN age < 35 THEN 'Young Adult (25-34)'
        WHEN age < 45 THEN 'Adult (35-44)'
        WHEN age < 55 THEN 'Middle Age (45-54)'
        ELSE '55+'
    END                                        AS age_group,
    COUNT(*)                                   AS burn_patients,
    COUNT_IF(sex = 'F')                        AS female,
    COUNT_IF(sex = 'M')                        AS male
FROM burn_patients
GROUP BY age_group
ORDER BY MIN(age);
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q8_mechanism() -> pd.DataFrame:
    """Q8.9 Fracture patients with a cause of injury recorded"""
    return _run("q8_mechanism", f"""
WITH src AS (
    SELECT unique_patient_id, visit_year, ALL_DIAGNOSES AS all_dx,
           '|' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(diagnosis_1_icd11, diagnosis_2_icd11, diagnosis_3_icd11, diagnosis_4_icd11, diagnosis_5_icd11, diagnosis_6_icd11, diagnosis_7_icd11, diagnosis_8_icd11), '|') AS codes,
           diagnosis_1, diagnosis_2, diagnosis_3, diagnosis_4, diagnosis_5, diagnosis_6, diagnosis_7, diagnosis_8
    FROM {TABLE}
    WHERE is_default_code_pair IS DISTINCT FROM TRUE
),
long AS (
    SELECT unique_patient_id, visit_year, all_dx, codes, diagnosis_1 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, all_dx, codes, diagnosis_2 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, all_dx, codes, diagnosis_3 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, all_dx, codes, diagnosis_4 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, all_dx, codes, diagnosis_5 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, all_dx, codes, diagnosis_6 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, all_dx, codes, diagnosis_7 AS dx FROM src
    UNION ALL
    SELECT unique_patient_id, visit_year, all_dx, codes, diagnosis_8 AS dx FROM src
),
fx AS (
    SELECT
        unique_patient_id, visit_year,
        BOOLOR_AGG(codes LIKE '%|P%')                                                                  AS m_external_code,
        BOOLOR_AGG(all_dx ILIKE ANY ('%land transport%', '%road traffic%', '%motor vehicle%', '%motorcycle%', '%pedestrian%')
                   AND all_dx NOT ILIKE '%assault%')                                                   AS m_transport,
        BOOLOR_AGG(REGEXP_LIKE(all_dx, '.*\\\\bfalls?\\\\b.*', 'i'))                                        AS m_fall,
        BOOLOR_AGG(all_dx ILIKE '%context of assault%' OR REGEXP_LIKE(all_dx, '.*\\\\bsport.*', 'i'))   AS m_assault_sport
    FROM long
    WHERE dx ILIKE '%fracture%' AND NOT (dx ILIKE ANY ('%union of fracture%', '%nontraumatic%', '%birth injury%'))
    GROUP BY unique_patient_id, visit_year
)
SELECT
    visit_year,
    COUNT(*)                                                                         AS fracture_patients,
    COUNT_IF(m_transport)                                                            AS with_transport_mechanism,
    COUNT_IF(m_fall)                                                                 AS with_fall_mechanism,
    COUNT_IF(m_assault_sport)                                                        AS with_assault_or_sport_mechanism,
    COUNT_IF(m_external_code)                                                        AS with_any_external_cause_code,
    ROUND(COUNT_IF(m_external_code) / NULLIF(COUNT(*), 0) * 100, 1)                  AS pct_with_external_cause_code
FROM fx
GROUP BY visit_year
ORDER BY visit_year;
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q8_sgbv_cat() -> pd.DataFrame:
    """Q8.5b SGBV patients by category, small cells suppressed"""
    return _run("q8_sgbv_cat", f"""
WITH sgbv AS (
    SELECT
        unique_patient_id,
        f.value::string AS context_code
    FROM {TABLE},
         LATERAL FLATTEN(INPUT => ARRAY_CONSTRUCT(
             diagnosis_1,diagnosis_2,diagnosis_3,diagnosis_4,
             diagnosis_5,diagnosis_6,diagnosis_7,diagnosis_8)) f
    WHERE f.value ILIKE '%context of assault%'
),
categorized AS (
    SELECT
        unique_patient_id,
        CASE
            WHEN context_code ILIKE '%sexual assault%'
              OR context_code ILIKE '%rape%'
              OR context_code ILIKE '%sodomy%'
              OR context_code ILIKE '%oral sex%'          THEN 'Sexual violence'
            WHEN context_code ILIKE '%neglect%'            THEN 'Neglect'
            WHEN context_code ILIKE '%torture%'
              OR context_code ILIKE '%robbery%'
              OR context_code ILIKE '%burglary%'
              OR context_code ILIKE '%retaliation%'
              OR context_code ILIKE '%bullying%'
              OR context_code ILIKE '%gang%'               THEN 'Physical violence'
            WHEN context_code ILIKE '%altercation%'
              OR context_code ILIKE '%blackmail%'
              OR context_code ILIKE '%misconduct%'         THEN 'Altercation/dispute'
            ELSE 'Other/unspecified'
        END                                        AS broad_category
    FROM sgbv
)
SELECT
    broad_category,
    IFF(COUNT(DISTINCT unique_patient_id) < 5, NULL, COUNT(DISTINCT unique_patient_id)) AS total_patients,
    (COUNT(DISTINCT unique_patient_id) < 5)                                             AS suppressed
FROM categorized
GROUP BY broad_category
UNION ALL
SELECT 'All SGBV patients', COUNT(DISTINCT unique_patient_id), FALSE
FROM categorized
ORDER BY total_patients DESC NULLS LAST;
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q8_sgbv_sexage() -> pd.DataFrame:
    """Q8.5 Sexual violence by sex and age band, small cells suppressed"""
    return _run("q8_sgbv_sexage", f"""
WITH sgbv AS (
    SELECT
        unique_patient_id, sex, age_in_years_float,
        f.value::string AS context_code
    FROM {TABLE},
         LATERAL FLATTEN(INPUT => ARRAY_CONSTRUCT(
             diagnosis_1,diagnosis_2,diagnosis_3,diagnosis_4,
             diagnosis_5,diagnosis_6,diagnosis_7,diagnosis_8)) f
    WHERE f.value ILIKE '%context of assault%'
),
categorized AS (
    SELECT
        unique_patient_id, sex, age_in_years_float,
        CASE
            WHEN context_code ILIKE '%sexual assault%'
              OR context_code ILIKE '%rape%'
              OR context_code ILIKE '%sodomy%'
              OR context_code ILIKE '%oral sex%'          THEN 'Sexual violence'
            WHEN context_code ILIKE '%neglect%'            THEN 'Neglect'
            WHEN context_code ILIKE '%torture%'
              OR context_code ILIKE '%robbery%'
              OR context_code ILIKE '%burglary%'
              OR context_code ILIKE '%retaliation%'
              OR context_code ILIKE '%bullying%'
              OR context_code ILIKE '%gang%'               THEN 'Physical violence'
            WHEN context_code ILIKE '%altercation%'
              OR context_code ILIKE '%blackmail%'
              OR context_code ILIKE '%misconduct%'         THEN 'Altercation/dispute'
            ELSE 'Other/unspecified'
        END                                        AS broad_category
    FROM sgbv
),
pat_cat AS (
    SELECT unique_patient_id, broad_category, MAX(sex) AS sex, MIN(age_in_years_float) AS age
    FROM categorized
    GROUP BY unique_patient_id, broad_category
),
cells AS (
    SELECT
        broad_category,
        sex,
        CASE
            WHEN age IS NULL THEN 'Age unknown'
            WHEN age < 18    THEN 'Minor (<18)'
            WHEN age < 35    THEN 'Adult (18-34)'
            ELSE '35+'
        END                                        AS age_band,
        COUNT(*)                                   AS n
    FROM pat_cat
    GROUP BY broad_category, sex, age_band
)
SELECT
    broad_category,
    sex,
    age_band,
    IFF(n < 5, NULL, n)                            AS patients,
    (n < 5)                                        AS suppressed
FROM cells
ORDER BY broad_category, age_band, sex;
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q8_sgbv_trend() -> pd.DataFrame:
    """Q8.7 Sexual-violence patients by year"""
    return _run("q8_sgbv_trend", f"""
WITH ctx AS (
    SELECT
        unique_patient_id, visit_year, age_in_years_float,
        f.value::string AS context_code
    FROM {TABLE},
         LATERAL FLATTEN(INPUT => ARRAY_CONSTRUCT(
             diagnosis_1,diagnosis_2,diagnosis_3,diagnosis_4,
             diagnosis_5,diagnosis_6,diagnosis_7,diagnosis_8)) f
    WHERE f.value ILIKE '%context of assault%'
),
py AS (
    SELECT
        unique_patient_id, visit_year,
        MIN(age_in_years_float) AS age,
        BOOLOR_AGG(context_code ILIKE '%sexual assault%'
                OR context_code ILIKE '%rape%'
                OR context_code ILIKE '%sodomy%'
                OR context_code ILIKE '%oral sex%') AS is_sexual
    FROM ctx
    GROUP BY unique_patient_id, visit_year
),
tot AS (
    SELECT visit_year, COUNT(DISTINCT unique_patient_id) AS all_patients
    FROM {TABLE}
    GROUP BY visit_year
)
SELECT
    t.visit_year,
    t.all_patients,
    COUNT(p.unique_patient_id)                          AS any_sgbv_patients,
    COUNT_IF(p.is_sexual)                               AS sexual_violence,
    COUNT_IF(p.is_sexual AND p.age < 18)                AS minors,
    COUNT_IF(p.is_sexual AND p.age >= 18)               AS adults
FROM tot t
LEFT JOIN py p ON p.visit_year = t.visit_year
GROUP BY t.visit_year, t.all_patients
ORDER BY t.visit_year;
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q8_sgbv_repeat() -> pd.DataFrame:
    """Q8.6 Repeat SGBV presentations"""
    return _run("q8_sgbv_repeat", f"""
WITH sgbv_patients AS (
    SELECT DISTINCT pno, unique_patient_id
    FROM {TABLE}
    WHERE ALL_DIAGNOSES ILIKE '%context of assault%'
)
SELECT
    COUNT(DISTINCT sp.unique_patient_id)                                    AS sgbv_patients,
    SUM(CASE WHEN visit_count > 1 THEN 1 ELSE 0 END)                        AS repeat_presentation_patients
FROM sgbv_patients sp
JOIN (
    SELECT unique_patient_id, COUNT(DISTINCT visit_year) AS visit_count
    FROM {TABLE}
    WHERE ALL_DIAGNOSES ILIKE '%context of assault%'
    GROUP BY unique_patient_id
) vc ON sp.unique_patient_id = vc.unique_patient_id;
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q8_sgbv_mh() -> pd.DataFrame:
    """Q8.8 SGBV patients with a mental-health code"""
    return _run("q8_sgbv_mh", f"""
WITH sgbv_patients AS (
    SELECT DISTINCT unique_patient_id
    FROM {TABLE}
    WHERE ALL_DIAGNOSES ILIKE '%context of assault%'
),
mh_patients AS (
    SELECT DISTINCT o.unique_patient_id
    FROM {TABLE} o,
         LATERAL FLATTEN(INPUT => ARRAY_CONSTRUCT(
             o.diagnosis_1_icd11, o.diagnosis_2_icd11, o.diagnosis_3_icd11, o.diagnosis_4_icd11,
             o.diagnosis_5_icd11, o.diagnosis_6_icd11, o.diagnosis_7_icd11, o.diagnosis_8_icd11
         )) f
    WHERE LEFT(f.value::string, 2) IN ('6A', '6B', '6C', '6D', '6E')
)
SELECT
    COUNT(DISTINCT sp.unique_patient_id)                                            AS total_sgbv_patients,
    COUNT(DISTINCT CASE WHEN mh.unique_patient_id IS NOT NULL
                        THEN sp.unique_patient_id END)                              AS with_any_mental_health_code
FROM sgbv_patients sp
LEFT JOIN mh_patients mh ON sp.unique_patient_id = mh.unique_patient_id;
""")


# ============================================================ Tab 9: Clinician workload (workload_queries.sql)
# Source table: HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD (Jan to Aug 2024). The duplicate rollup row is excluded in every query.


@st.cache_data(ttl=3600, show_spinner=False)
def q9_summary() -> pd.DataFrame:
    """Q9.1 Totals and distribution, duplicate row excluded"""
    return _run("q9_summary", f"""
SELECT
    COUNT(*)                                            AS total_clinicians,
    SUM(CASE WHEN total_contacts > 0 THEN 1 ELSE 0 END) AS active_clinicians,
    SUM(CASE WHEN total_contacts = 0 THEN 1 ELSE 0 END) AS zero_load_clinicians,
    SUM(total_contacts)                                 AS total_contacts,
    ROUND(AVG(CASE WHEN total_contacts > 0
              THEN total_contacts END), 0)              AS mean_contacts_active,
    MEDIAN(CASE WHEN total_contacts > 0
                THEN total_contacts END)                AS median_contacts_active,
    MAX(total_contacts)                                 AS max_contacts
FROM (
    SELECT
        clinician_id,
        COALESCE(jan_2024,0) + COALESCE(feb_2024,0) + COALESCE(mar_2024,0) +
        COALESCE(apr_2024,0) + COALESCE(may_2024,0) + COALESCE(jun_2024,0) +
        COALESCE(jul_2024,0) + COALESCE(aug_2024,0)    AS total_contacts
    FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    WHERE clinician_id != 'ab2882c86465b1daf5745ec9617b8093dcedf720cddc978f715fafa6f0d91bc9'
)
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q9_top() -> pd.DataFrame:
    """Q9.4 Top 15 clinicians with monthly contacts and the upper fence"""
    return _run("q9_top", f"""
WITH totals AS (
    SELECT
        *,
        COALESCE(jan_2024,0)+COALESCE(feb_2024,0)+COALESCE(mar_2024,0)+
        COALESCE(apr_2024,0)+COALESCE(may_2024,0)+COALESCE(jun_2024,0)+
        COALESCE(jul_2024,0)+COALESCE(aug_2024,0)   AS total_contacts,
        (COALESCE(jan_2024,0) > 0)::INT + (COALESCE(feb_2024,0) > 0)::INT +
        (COALESCE(mar_2024,0) > 0)::INT + (COALESCE(apr_2024,0) > 0)::INT +
        (COALESCE(may_2024,0) > 0)::INT + (COALESCE(jun_2024,0) > 0)::INT +
        (COALESCE(jul_2024,0) > 0)::INT + (COALESCE(aug_2024,0) > 0)::INT AS months_active
    FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    WHERE clinician_id != 'ab2882c86465b1daf5745ec9617b8093dcedf720cddc978f715fafa6f0d91bc9'
),
fence AS (
    SELECT
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_contacts)
            + 1.5 * (PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_contacts)
                   - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_contacts)), 0) AS upper_fence
    FROM totals
    WHERE total_contacts > 0
)
SELECT
    t.*,
    f.upper_fence,
    t.total_contacts > f.upper_fence AS is_statistical_outlier
FROM totals t
CROSS JOIN fence f
ORDER BY t.total_contacts DESC
LIMIT 15
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q9_concentration() -> pd.DataFrame:
    """Q9.3 Top 10 clinicians vs the rest"""
    return _run("q9_concentration", f"""
WITH totals AS (
    SELECT
        clinician_id,
        COALESCE(jan_2024,0)+COALESCE(feb_2024,0)+COALESCE(mar_2024,0)+
        COALESCE(apr_2024,0)+COALESCE(may_2024,0)+COALESCE(jun_2024,0)+
        COALESCE(jul_2024,0)+COALESCE(aug_2024,0)   AS total_contacts
    FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    WHERE clinician_id != 'ab2882c86465b1daf5745ec9617b8093dcedf720cddc978f715fafa6f0d91bc9'
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (ORDER BY total_contacts DESC, clinician_id) AS rnk
    FROM totals
)
SELECT
    CASE WHEN rnk <= 10 THEN 'Top 10 clinicians' ELSE 'Remaining clinicians' END AS group_label,
    COUNT(*)                                            AS clinician_count,
    SUM(total_contacts)                                 AS contacts,
    ROUND(SUM(total_contacts) * 100.0 / SUM(SUM(total_contacts)) OVER (), 1) AS pct_of_total
FROM ranked
GROUP BY group_label
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q9_monthly_outliers() -> pd.DataFrame:
    """Q9.6 Monthly outlier detail with sustained vs one-off pattern"""
    return _run("q9_monthly_outliers", f"""
WITH monthly AS (
    SELECT clinician_id, 'Jan' AS month, 1 AS month_order, jan_2024 AS contacts
    FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    UNION ALL
    SELECT clinician_id, 'Feb', 2, feb_2024 FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    UNION ALL
    SELECT clinician_id, 'Mar', 3, mar_2024 FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    UNION ALL
    SELECT clinician_id, 'Apr', 4, apr_2024 FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    UNION ALL
    SELECT clinician_id, 'May', 5, may_2024 FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    UNION ALL
    SELECT clinician_id, 'Jun', 6, jun_2024 FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    UNION ALL
    SELECT clinician_id, 'Jul', 7, jul_2024 FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    UNION ALL
    SELECT clinician_id, 'Aug', 8, aug_2024 FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
),
real_monthly AS (
    SELECT * FROM monthly
    WHERE clinician_id != 'ab2882c86465b1daf5745ec9617b8093dcedf720cddc978f715fafa6f0d91bc9'
),
active_monthly AS (
    SELECT * FROM real_monthly WHERE COALESCE(contacts, 0) > 0
),
monthly_quartiles AS (
    SELECT
        month, month_order,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY contacts) AS q1,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY contacts) AS q3
    FROM active_monthly
    GROUP BY month, month_order
),
flagged AS (
    SELECT
        am.clinician_id, am.month, am.month_order,
        am.contacts > mq.q3 + 1.5 * (mq.q3 - mq.q1) AS is_flagged_month
    FROM active_monthly am
    JOIN monthly_quartiles mq ON am.month = mq.month
),
per_clinician AS (
    SELECT clinician_id, COUNT_IF(is_flagged_month) AS months_flagged
    FROM flagged
    GROUP BY clinician_id
    HAVING COUNT_IF(is_flagged_month) > 0
)
SELECT
    rm.month,
    rm.month_order,
    rm.clinician_id,
    COALESCE(rm.contacts, 0)                                AS contacts,
    ROUND(mq.q3 + 1.5 * (mq.q3 - mq.q1), 0)                 AS upper_fence,
    COALESCE(rm.contacts, 0) > mq.q3 + 1.5 * (mq.q3 - mq.q1) AS is_flagged_month,
    pc.months_flagged,
    CASE WHEN pc.months_flagged >= 4 THEN 'Sustained'
         WHEN pc.months_flagged >= 2 THEN 'Occasional'
         ELSE 'Single month' END                            AS pattern
FROM real_monthly rm
JOIN per_clinician pc     ON rm.clinician_id = pc.clinician_id
JOIN monthly_quartiles mq ON rm.month = mq.month
ORDER BY pc.months_flagged DESC, rm.clinician_id, rm.month_order
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q9_monthly() -> pd.DataFrame:
    """Q9.7 Contacts and active clinicians by month"""
    return _run("q9_monthly", f"""
WITH monthly AS (
    SELECT clinician_id, 'Jan 2024' AS month_name, 1 AS month_order, jan_2024 AS contacts
    FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    UNION ALL
    SELECT clinician_id, 'Feb 2024', 2, feb_2024 FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    UNION ALL
    SELECT clinician_id, 'Mar 2024', 3, mar_2024 FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    UNION ALL
    SELECT clinician_id, 'Apr 2024', 4, apr_2024 FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    UNION ALL
    SELECT clinician_id, 'May 2024', 5, may_2024 FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    UNION ALL
    SELECT clinician_id, 'Jun 2024', 6, jun_2024 FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    UNION ALL
    SELECT clinician_id, 'Jul 2024', 7, jul_2024 FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    UNION ALL
    SELECT clinician_id, 'Aug 2024', 8, aug_2024 FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
)
SELECT
    month_name,
    month_order,
    COUNT_IF(COALESCE(contacts, 0) > 0)                         AS active_clinicians,
    SUM(COALESCE(contacts, 0))                                  AS total_contacts,
    ROUND(SUM(COALESCE(contacts, 0))
          / NULLIF(COUNT_IF(COALESCE(contacts, 0) > 0), 0), 1)  AS contacts_per_active_clinician,
    MEDIAN(CASE WHEN COALESCE(contacts, 0) > 0 THEN contacts END) AS median_contacts_per_active_clinician
FROM monthly
WHERE clinician_id != 'ab2882c86465b1daf5745ec9617b8093dcedf720cddc978f715fafa6f0d91bc9'
GROUP BY month_name, month_order
ORDER BY month_order
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q9_bands() -> pd.DataFrame:
    """Q9.8 Clinicians by 8-month contact band"""
    return _run("q9_bands", f"""
WITH totals AS (
    SELECT
        clinician_id,
        COALESCE(jan_2024,0)+COALESCE(feb_2024,0)+COALESCE(mar_2024,0)+
        COALESCE(apr_2024,0)+COALESCE(may_2024,0)+COALESCE(jun_2024,0)+
        COALESCE(jul_2024,0)+COALESCE(aug_2024,0)   AS total_contacts
    FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    WHERE clinician_id != 'ab2882c86465b1daf5745ec9617b8093dcedf720cddc978f715fafa6f0d91bc9'
),
banded AS (
    SELECT
        total_contacts,
        CASE WHEN total_contacts = 0    THEN '0 (no contacts)'
             WHEN total_contacts < 100  THEN '1-99'
             WHEN total_contacts < 300  THEN '100-299'
             WHEN total_contacts < 1000 THEN '300-999'
             ELSE '1,000+' END                      AS contact_band,
        CASE WHEN total_contacts = 0    THEN 1
             WHEN total_contacts < 100  THEN 2
             WHEN total_contacts < 300  THEN 3
             WHEN total_contacts < 1000 THEN 4
             ELSE 5 END                             AS band_order
    FROM totals
)
SELECT
    contact_band,
    band_order,
    COUNT(*)                                        AS clinicians,
    SUM(total_contacts)                             AS contacts,
    ROUND(SUM(total_contacts) * 100.0 / SUM(SUM(total_contacts)) OVER (), 1) AS pct_of_contacts
FROM banded
GROUP BY contact_band, band_order
ORDER BY band_order
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q9_months_active() -> pd.DataFrame:
    """Q9.9 Clinicians by months active"""
    return _run("q9_months_active", f"""
WITH per_clinician AS (
    SELECT
        clinician_id,
        (COALESCE(jan_2024,0) > 0)::INT + (COALESCE(feb_2024,0) > 0)::INT +
        (COALESCE(mar_2024,0) > 0)::INT + (COALESCE(apr_2024,0) > 0)::INT +
        (COALESCE(may_2024,0) > 0)::INT + (COALESCE(jun_2024,0) > 0)::INT +
        (COALESCE(jul_2024,0) > 0)::INT + (COALESCE(aug_2024,0) > 0)::INT AS months_active,
        COALESCE(jan_2024,0)+COALESCE(feb_2024,0)+COALESCE(mar_2024,0)+
        COALESCE(apr_2024,0)+COALESCE(may_2024,0)+COALESCE(jun_2024,0)+
        COALESCE(jul_2024,0)+COALESCE(aug_2024,0)   AS total_contacts
    FROM HOSPITALS.NANDI_CLEAN.CLINICIANS_WORKLOAD
    WHERE clinician_id != 'ab2882c86465b1daf5745ec9617b8093dcedf720cddc978f715fafa6f0d91bc9'
),
spine AS (
    SELECT column1 AS months_active FROM VALUES (0),(1),(2),(3),(4),(5),(6),(7),(8)
)
SELECT
    s.months_active,
    COUNT(p.clinician_id)                           AS clinicians,
    COALESCE(SUM(p.total_contacts), 0)              AS contacts
FROM spine s
LEFT JOIN per_clinician p ON p.months_active = s.months_active
GROUP BY s.months_active
ORDER BY s.months_active
""")


# ============================================================ Tab 11: Data quality (dataquality_queries.sql)


@st.cache_data(ttl=3600, show_spinner=False)
def q11_diag() -> pd.DataFrame:
    """Q11.1 Diagnosis coverage summary"""
    return _run("q11_diag", f"""
SELECT
    COUNT(*)                                            AS total_visits,
    SUM(CASE WHEN diagnosis_1 IS NULL THEN 1 ELSE 0 END) AS no_diagnosis,
    ROUND(SUM(CASE WHEN diagnosis_1 IS NULL THEN 1.0 ELSE 0.0 END)
          / COUNT(*) * 100, 1)                          AS pct_no_diagnosis,
    SUM(CASE WHEN diagnoses_unresolved = TRUE THEN 1 ELSE 0 END) AS unresolved_dx,
    ROUND(SUM(CASE WHEN diagnoses_unresolved = TRUE THEN 1.0 ELSE 0.0 END)
          / COUNT(*) * 100, 1)                          AS pct_unresolved,
    SUM(CASE WHEN has_truncated_diagnosis = TRUE THEN 1 ELSE 0 END) AS truncated_dx,
    SUM(CASE WHEN is_default_code_pair = TRUE THEN 1 ELSE 0 END)  AS aaa_default_pair
FROM HOSPITALS.NANDI_CLEAN.NANDI_OUTPATIENT_WIDE
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q11_geo() -> pd.DataFrame:
    """Q11.2 Residence (geo) confidence by county"""
    return _run("q11_geo", f"""
SELECT
    geo_confidence,
    county,
    COUNT(*)                                            AS visits,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
FROM HOSPITALS.NANDI_CLEAN.NANDI_OUTPATIENT_WIDE
GROUP BY geo_confidence, county
ORDER BY visits DESC
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q11_year() -> pd.DataFrame:
    """Q11.6 Year completeness with the 2025 export-gap flag"""
    return _run("q11_year", f"""
SELECT
    visit_year,
    COUNT(*)                                            AS visits,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct_of_total,
    ROUND(COUNT(*) * 100.0 / MAX(COUNT(*)) OVER (), 1) AS pct_of_busiest_year,
    CASE
        WHEN visit_year IN (2023, 2026) THEN 'PARTIAL — exclude from YoY comparisons'
        WHEN visit_year = 2025          THEN 'COMPLETE YEAR, KNOWN EXPORT GAP — compare with care'
        ELSE 'Full year — comparable'
    END                                                 AS completeness_flag
FROM HOSPITALS.NANDI_CLEAN.NANDI_OUTPATIENT_WIDE
GROUP BY visit_year
ORDER BY visit_year
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q11_tracking() -> pd.DataFrame:
    """Q11.3 Tracking classification"""
    return _run("q11_tracking", f"""
SELECT
    tracking_classification,
    COUNT(*)                                            AS visits,
    COUNT(DISTINCT unique_patient_id)                   AS unique_patients,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
FROM HOSPITALS.NANDI_CLEAN.NANDI_OUTPATIENT_WIDE
GROUP BY tracking_classification
ORDER BY visits DESC
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q11_dupes() -> pd.DataFrame:
    """Q11.4 Possible duplicate flag"""
    return _run("q11_dupes", f"""
SELECT
    possible_duplicate_flag,
    COUNT(*)                                            AS visits,
    COUNT(DISTINCT unique_patient_id)                   AS unique_patients
FROM HOSPITALS.NANDI_CLEAN.NANDI_OUTPATIENT_WIDE
GROUP BY possible_duplicate_flag
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q11_gender() -> pd.DataFrame:
    """Q11.5 Gender disparity flag"""
    return _run("q11_gender", f"""
SELECT
    gender_disparity_flag,
    COUNT(DISTINCT unique_patient_id)                   AS patients,
    COUNT(*)                                            AS visits
FROM HOSPITALS.NANDI_CLEAN.NANDI_OUTPATIENT_WIDE
GROUP BY gender_disparity_flag
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q11_malnutrition() -> pd.DataFrame:
    """DQ note: malnutrition coding volume (feeds the register)"""
    return _run("q11_malnutrition", f"""
SELECT
    CASE WHEN age_in_years_float < 5 THEN 'Under-5' ELSE '5+' END AS age_band,
    COUNT(*) AS visits_with_malnutrition_code
FROM HOSPITALS.NANDI_CLEAN.NANDI_OUTPATIENT_WIDE
WHERE ALL_DIAGNOSES ILIKE ANY (
    '%malnutrition%','%wasting%','%stunting%','%undernutrition%',
    '%kwashiorkor%','%marasmus%','%underweight%','%failure to thrive%'
)
GROUP BY age_band
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q11_complete() -> pd.DataFrame:
    """Q11.7 Reporting completeness by year"""
    return _run("q11_complete", f"""
SELECT
    visit_year,
    COUNT(*)                                                         AS visits,
    COUNT_IF(diagnosis_1 IS NULL)                                    AS no_diagnosis,
    ROUND(COUNT_IF(diagnosis_1 IS NULL) * 100.0 / COUNT(*), 1)       AS pct_no_diagnosis,
    COUNT_IF(diagnosis_1 IS NOT NULL)                                AS coded_visits,
    COUNT_IF(diagnosis_2 IS NOT NULL)                                AS multi_diagnosis_visits,
    ROUND(COUNT_IF(diagnosis_2 IS NOT NULL) * 100.0
          / NULLIF(COUNT_IF(diagnosis_1 IS NOT NULL), 0), 1)         AS pct_multi_diagnosis_of_coded,
    COUNT_IF(has_truncated_diagnosis = TRUE)                         AS truncated_visits,
    ROUND(COUNT_IF(has_truncated_diagnosis = TRUE) * 100.0 / COUNT(*), 1) AS pct_truncated
FROM HOSPITALS.NANDI_CLEAN.NANDI_OUTPATIENT_WIDE
GROUP BY visit_year
ORDER BY visit_year
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q11_ward_match() -> pd.DataFrame:
    """Q11.8 Patients whose ward has no boundary match"""
    return _run("q11_ward_match", f"""
WITH geo AS (
    SELECT DISTINCT UPPER(TRIM(geo_name)) AS ward_key, UPPER(TRIM(parent_name)) AS sub_county_key
    FROM HOSPITALS.LOOKUP_TABLES.NANDI_GEO_COORDINATES
    WHERE geo_level = 'WARD'
      AND geocode_status = 'FOUND'
      AND county ILIKE 'Nandi'
),
visits AS (
    SELECT
        v.unique_patient_id,
        (g.ward_key IS NOT NULL) AS has_boundary
    FROM HOSPITALS.NANDI_CLEAN.NANDI_OUTPATIENT_WIDE v
    LEFT JOIN geo g
           ON g.ward_key       = UPPER(TRIM(v.ward))
          AND g.sub_county_key = UPPER(TRIM(v.sub_county))
    WHERE v.county = 'Nandi'
      AND v.ward IS NOT NULL
)
SELECT
    CASE WHEN has_boundary THEN 'Ward matches a boundary' ELSE 'No boundary match' END AS match_status,
    COUNT(*)                                            AS visits,
    COUNT(DISTINCT unique_patient_id)                   AS patients,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct_of_visits
FROM visits
GROUP BY has_boundary
ORDER BY has_boundary DESC
""")


@st.cache_data(ttl=3600, show_spinner=False)
def q11_ward_list() -> pd.DataFrame:
    """Q11.8b Wards that do not match a boundary"""
    return _run("q11_ward_list", f"""
WITH geo AS (
    SELECT DISTINCT UPPER(TRIM(geo_name)) AS ward_key, UPPER(TRIM(parent_name)) AS sub_county_key
    FROM HOSPITALS.LOOKUP_TABLES.NANDI_GEO_COORDINATES
    WHERE geo_level = 'WARD'
      AND geocode_status = 'FOUND'
      AND county ILIKE 'Nandi'
)
SELECT
    v.ward,
    v.sub_county,
    COUNT(*)                            AS visits,
    COUNT(DISTINCT v.unique_patient_id) AS patients
FROM HOSPITALS.NANDI_CLEAN.NANDI_OUTPATIENT_WIDE v
LEFT JOIN geo g
       ON g.ward_key       = UPPER(TRIM(v.ward))
      AND g.sub_county_key = UPPER(TRIM(v.sub_county))
WHERE v.county = 'Nandi'
  AND v.ward IS NOT NULL
  AND g.ward_key IS NULL
GROUP BY v.ward, v.sub_county
ORDER BY patients DESC
LIMIT 20
""")



# ============================================================ TAB 1 · SUMMARY (summary_queries.sql)
@st.cache_data(ttl=3600, show_spinner=False)
def q1_year() -> pd.DataFrame:
    """Q1.2 (modified) Visits by year; the AAA default code pair rows are excluded so the years add up to Q1.1."""
    return _run("q1_year", f"""
    SELECT visit_year,
           COUNT(*)                                          AS visits,
           COUNT(DISTINCT unique_patient_id)                 AS unique_patients,
           SUM(CASE WHEN sex = 'F' THEN 1 ELSE 0 END)        AS female_visits,
           SUM(CASE WHEN sex = 'M' THEN 1 ELSE 0 END)        AS male_visits,
           ROUND(SUM(CASE WHEN sex = 'F' THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 1) AS pct_female
    FROM {TABLE}
    WHERE {NOT_DEFAULT_PAIR}
    GROUP BY visit_year
    ORDER BY visit_year
    """)
