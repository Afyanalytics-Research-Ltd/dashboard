"""Snowflake connectivity — RSA key-pair auth, read-only usage.

WHY this shape:

- The account's MFA policy blocks programmatic password auth, so the only
  supported path is RSA key-pair (private key PEM → DER passed to the
  connector). No passwords, and no secrets ever printed or logged.
- ``snowflake.connector`` and ``cryptography`` are imported *lazily* inside
  functions: offline runs must work on machines without either package
  installed.
- Connection identity comes from environment variables with documented
  defaults; the private-key location comes from ``SNOWFLAKE_PRIVATE_KEY_PATH``
  with a fallback to the workspace-root ``rsa_key.p8``. The fallback is
  resolved relative to *this package's own root* (two levels up from this
  file) — deliberately not the cross-repo ``parents[5]`` reach the KSH
  reference module used, which broke whenever the repo layout shifted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

# Workspace root = the directory that contains the inventory_intelligence
# package (".../st peter's orthopedic"). parents[0]=utils, [1]=package, [2]=root.
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

#: Environment-variable defaults for the SPH warehouse.
#: Every value is overridable via the same-named environment variable.
ENV_DEFAULTS: Mapping[str, str] = {
    "SNOWFLAKE_ACCOUNT": "UFLYZNZ-RA32706",
    "SNOWFLAKE_USER": "SAMUEL.SEKA",
    "SNOWFLAKE_ROLE": "DATAANALYSTS",
    "SNOWFLAKE_WAREHOUSE": "COMPUTE_WH",
    "SNOWFLAKE_DATABASE": "HOSPITALS",
}


def _load_dotenv_if_available() -> None:
    """Best-effort .env loading (package root, then workspace root).

    Optional: python-dotenv is in requirements but its absence must never
    break imports (offline tests).
    """
    try:
        from dotenv import load_dotenv  # lazy — optional convenience only
    except ImportError:
        return
    for candidate in (_WORKSPACE_ROOT / ".env", Path(__file__).resolve().parents[1] / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)


def _setting(name: str) -> str:
    value = os.getenv(name, ENV_DEFAULTS.get(name, ""))
    return value.strip()


def default_private_key_path() -> Path:
    """Resolve the RSA private-key PEM path.

    Order: ``SNOWFLAKE_PRIVATE_KEY_PATH`` env var, else ``rsa_key.p8`` in the
    workspace root next to this package.
    """
    env_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return _WORKSPACE_ROOT / "rsa_key.p8"


def _load_private_key_der(key_path: Path) -> bytes:
    """Load a PEM (PKCS#8) private key and re-serialize to unencrypted DER.

    The Snowflake connector's ``private_key`` argument takes DER bytes.
    An optional passphrase is read from ``SNOWFLAKE_PRIVATE_KEY_PASSPHRASE``.
    The key material is never logged, printed, or attached to exceptions.
    """
    from cryptography.hazmat.primitives import serialization  # lazy

    passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "")
    pem_bytes = Path(key_path).read_bytes()
    private_key = serialization.load_pem_private_key(
        pem_bytes,
        password=passphrase.encode() if passphrase else None,
    )
    return private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def get_connection(key_path: Optional[Path] = None) -> Any:
    """Open a read-only-intent Snowflake connection using RSA key-pair auth.

    Returns a ``snowflake.connector.SnowflakeConnection``. Typed as ``Any``
    so this module can be imported (and everything else tested) without the
    connector installed.
    """
    _load_dotenv_if_available()
    import snowflake.connector  # lazy — never required for offline runs

    return snowflake.connector.connect(
        account=_setting("SNOWFLAKE_ACCOUNT"),
        user=_setting("SNOWFLAKE_USER"),
        role=_setting("SNOWFLAKE_ROLE"),
        warehouse=_setting("SNOWFLAKE_WAREHOUSE"),
        database=_setting("SNOWFLAKE_DATABASE"),
        private_key=_load_private_key_der(key_path or default_private_key_path()),
        session_parameters={"QUERY_TAG": "inventory_intelligence"},
    )


def run_query(
    conn: Any,
    sql: str,
    params: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Execute ``sql`` with optional bind ``params`` and return a DataFrame.

    - Bind parameters use the connector's ``pyformat`` style
      (``%(name)s`` + dict), keeping values out of SQL text
      (``data/queries.py`` builders emit this style).
    - Prefers ``cursor.fetch_pandas_all()`` (Arrow fast path); falls back to
      ``fetchall()`` + ``cursor.description`` when the Arrow result path is
      unavailable for a statement type.
    - Column names are normalized to lowercase so downstream code has one
      casing convention regardless of fetch path.
    """
    cur = conn.cursor()
    try:
        cur.execute(sql, params)
        try:
            df = cur.fetch_pandas_all()
        except Exception:
            columns = [d[0] for d in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=columns)
    finally:
        cur.close()
    df.columns = [str(c).lower() for c in df.columns]
    return df
