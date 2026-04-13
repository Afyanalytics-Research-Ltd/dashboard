"""
config.py  —  Xana Pipeline
══════════════════════════════════════════════════════════════════════════════
SNOWFLAKE STRUCTURE (source)
─────────────────────────────
Database : HOSPITALS
Schemas  : TENRI_RAW, KAKAMEGA_RAW, LODWAR_RAW, KISUMU_RAW
Table    : EVENTS_RAW  (one per schema)

Fully-qualified reference example:
    HOSPITALS.TENRI_RAW.EVENTS_RAW

MYSQL STRUCTURE (destination)
──────────────────────────────
Database : HOSPITALS   (mirrors the Snowflake database name)
Schemas  : TENRI_RAW   → raw landing zone for tenri source
           KAKAMEGA_RAW→ raw landing zone for kakamega source
           LODWAR_RAW  → raw landing zone for lodwar source
           KISUMU_RAW  → raw landing zone for kisumu source
           STAGING     → unified staging with surrogate keys (all sources)
           REPORTING   → conformed dims + facts + NLP mart (Power BI reads this)

ADDING A NEW SOURCE SCHEMA
──────────────────────────
1. Add one entry to SNOWFLAKE_SOURCES:
       "nairobi": "NAIROBI_RAW"
2. Run: python run_pipeline.py --schema nairobi --full
══════════════════════════════════════════════════════════════════════════════
"""

import os

# ── MySQL connection ──────────────────────────────────────────────────────────
DB_USER = os.getenv("DB_USER",   "root")
DB_PASS = os.getenv("DB_PASS",   "ie97#")
DB_HOST = os.getenv("DB_HOST",   "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))

# ── MySQL database ─────────────────────────────────────────────────────────────
# Single database — mirrors Snowflake's HOSPITALS database.
MYSQL_DATABASE = "HOSPITALS"

# ── MySQL schema names ─────────────────────────────────────────────────────────
# Each source gets its own raw schema. Naming mirrors Snowflake exactly.
# Unified staging and reporting are shared across all sources.
STAGING_SCHEMA   = "TENRI_RAW"  # staging tables (STG_ prefix) live here
REPORTING_SCHEMA = "TENRI_RAW"  # reporting tables (REPORTING_ prefix) live here

# Aliases used by run_pipeline.py
STAGING_DB   = "TENRI_RAW"
REPORTING_DB = "TENRI_RAW"
RAW_DB       = "TENRI_RAW"


def raw_schema(logical_name: str) -> str:
    """Return the MySQL raw schema name for a given logical source.
    e.g. raw_schema("tenri") → "TENRI_RAW"
    """
    return SNOWFLAKE_SOURCES[logical_name]


def fq(schema: str, table: str) -> str:
    """Fully-qualified MySQL table reference: `HOSPITALS`.`TENRI_RAW`.`table`"""
    return f"`{MYSQL_DATABASE}`.`{schema}`.`{table}`"


# ── Snowflake connection ──────────────────────────────────────────────────────
SF_ACCOUNT       = os.getenv("SF_ACCOUNT",       "")
SF_USER          = os.getenv("SF_USER",           "")
SF_PASSWORD      = os.getenv("SF_PASSWORD",       "")
SF_ROLE          = os.getenv("SF_ROLE",           "DATAANALYSTS")
SF_WAREHOUSE     = os.getenv("SF_WAREHOUSE",      "COMPUTE_WH")
SF_DATABASE      = os.getenv("SF_DATABASE",       "HOSPITALS")
SF_AUTHENTICATOR = os.getenv("SF_AUTHENTICATOR",  "username_password_mfa")
SF_EVENTS_SCHEMA = os.getenv("SF_EVENTS_SCHEMA",  "TENRI_RAW")

# ── Source schemas ────────────────────────────────────────────────────────────
# Maps logical name → Snowflake schema name.
# The same mapping is used for MySQL raw schema names.
# Logical name: used in surrogate keys  →  "tenri|42"
# SF/MySQL schema: TENRI_RAW           →  HOSPITALS.TENRI_RAW.EVENTS_RAW
SNOWFLAKE_SOURCES = {
    "tenri":    "TENRI_RAW",
    #"kakamega": "KAKAMEGA_RAW",
    #"lodwar":   "LODWAR_RAW",
    #"kisumu":   "KISUMU_RAW",
}


def sf_events_raw(sf_schema: str) -> str:
    """Fully-qualified Snowflake EVENTS_RAW reference.
    e.g. sf_events_raw("TENRI_RAW") → "HOSPITALS.TENRI_RAW.EVENTS_RAW"
    """
    return f"{SF_DATABASE}.{sf_schema}.EVENTS_RAW"


# ── Date floor ────────────────────────────────────────────────────────────────
DATA_CUTOFF_DATE = "2018-01-01"

# ── Chunk size for mart rebuilds ──────────────────────────────────────────────
CHUNK_SIZE = 5_000