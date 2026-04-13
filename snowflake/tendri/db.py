"""
db.py  —  Xana Snowflake Pipeline
Database connection helpers.
"""

import configparser
import os
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

import snowflake.connector  # type: ignore
from config import (
    DB_USER, DB_PASS, DB_HOST, DB_PORT,
    SF_ACCOUNT, SF_USER, SF_PASSWORD,
    SF_ROLE, SF_WAREHOUSE, SF_DATABASE, SF_EVENTS_SCHEMA,
    SF_AUTHENTICATOR,
)


def build_mysql_engine(database: str = "") -> Engine:
    """
    Return a SQLAlchemy engine connected to MySQL.
    database="" connects without a default database (used for CREATE DATABASE).
    """
    url = (
        f"mysql+pymysql://{DB_USER}:{DB_PASS}"
        f"@{DB_HOST}:{DB_PORT}/{database}"
        f"?charset=utf8mb4"
    )
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)


def build_snowflake_conn(totp: Optional[str] = None):
    """
    Return a Snowflake connector connection.

    Credentials are read from:
      1. Environment variables (SF_ACCOUNT, SF_USER, etc.)
      2. config.ini [snowflake] section as fallback

    totp — Duo MFA passcode. Pass None for service accounts (no MFA).
    """
    # Config.ini fallback
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(os.path.dirname(__file__), "..", "config.ini"))
    sf_cfg = cfg["snowflake"] if "snowflake" in cfg else None

    def _get(env_key: str, ini_key: str, default: str = "") -> str:
        env_val = os.getenv(env_key, "")
        if env_val:
            return env_val
        if sf_cfg and ini_key in sf_cfg:
            return sf_cfg[ini_key]
        return default

    token = _get("SF_TOKEN", "token")

    if token:
        # Programmatic Access Token — no password or TOTP needed
        return snowflake.connector.connect(
            account       = _get("SF_ACCOUNT",   "account"),
            user          = _get("SF_USER",       "user"),
            authenticator = "oauth",
            token         = token,
            role          = _get("SF_ROLE",      "role",      "DATAANALYSTS"),
            warehouse     = _get("SF_WAREHOUSE", "warehouse", "COMPUTE_WH"),
            database      = _get("SF_DATABASE",  "database",  "HOSPITALS"),
            schema        = _get("SF_EVENTS_SCHEMA", "schema", "TENRI_RAW"),
        )

    kwargs = {
        "account":    _get("SF_ACCOUNT",   "account"),
        "user":       _get("SF_USER",      "user"),
        "password":   _get("SF_PASSWORD",  "password"),
        "role":       _get("SF_ROLE",      "role",      "DATAANALYSTS"),
        "warehouse":  _get("SF_WAREHOUSE", "warehouse", "COMPUTE_WH"),
        "database":   _get("SF_DATABASE",  "database",  "HOSPITALS"),
        "schema":     _get("SF_EVENTS_SCHEMA", "schema", "TENRI_RAW"),
    }

    authenticator = _get("SF_AUTHENTICATOR", "authenticator", "snowflake")
    if authenticator != "snowflake":
        kwargs["authenticator"] = authenticator

    if totp:
        kwargs["passcode"] = totp

    return snowflake.connector.connect(**kwargs)