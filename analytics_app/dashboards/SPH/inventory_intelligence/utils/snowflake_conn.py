"""Snowflake connectivity
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

# _REPO_ROOT = the dashboard repo root (main-repo layout); _WORKSPACE_ROOT keeps
# the standalone-repo layout working.
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
try:
    _REPO_ROOT = Path(__file__).resolve().parents[5]
except IndexError:
    _REPO_ROOT = _WORKSPACE_ROOT


def _load_dotenv_if_available() -> None:
  
    try:
        from dotenv import load_dotenv  
    except ImportError:
        return
    for candidate in (_REPO_ROOT / ".env", _WORKSPACE_ROOT / ".env",
                      Path(__file__).resolve().parents[1] / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)


def _setting(name: str) -> str:
    return os.getenv(name, "").strip()


def default_private_key_path() -> Path:
    """RSA key path: ``SNOWFLAKE_PRIVATE_KEY_PATH`` (a relative value is anchored
    to the repo root, like KSH), else ``rsa_key.p8`` at the repo/workspace root."""
    env_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "").strip()
    if env_path:
        p = Path(env_path).expanduser()
        return p if p.is_absolute() else (_REPO_ROOT / p)
    for root in (_REPO_ROOT, _WORKSPACE_ROOT):
        if (root / "rsa_key.p8").is_file():
            return root / "rsa_key.p8"
    return _REPO_ROOT / "rsa_key.p8"


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
