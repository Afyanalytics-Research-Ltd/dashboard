"""Snowflake access for the V3 dashboard.

Credentials: the repo's existing `.env` (key-pair auth). The `.env` is located by walking up from this
file, so the app works from the main tree and from any worktree nested inside it; a relative
SNOWFLAKE_PRIVATE_KEY_PATH is resolved against the `.env`'s folder (never the working directory).
Nothing secret is logged or displayed. Role: the user's default role (read-only use of REPORTING).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import streamlit as st

_TOKEN_EXPIRED = 390114


def _env_file() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def _load_env() -> Path | None:
    env = _env_file()
    if env:
        from dotenv import load_dotenv
        load_dotenv(env, override=False)
    return env


@st.cache_resource(show_spinner=False)
def _connection():
    import snowflake.connector

    env = _load_env()
    key_path = Path(os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"].strip())
    if not key_path.is_absolute() and env is not None:
        key_path = (env.parent / key_path).resolve()
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"].strip(),
        user=os.environ["SNOWFLAKE_USER"].strip(),
        private_key_file=str(key_path),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "").strip() or None,
        database="HOSPITALS",
        login_timeout=30,
        network_timeout=180,
        client_session_keep_alive=True,
        session_parameters={"STATEMENT_TIMEOUT_IN_SECONDS": 120,
                            "QUERY_TAG": "ksh_v3_inventory_dashboard"},
    )


def query(sql: str, params: tuple | None = None) -> pd.DataFrame:
    """Run one read query; lower-case column names. Reconnects once on an expired session."""
    for attempt in (1, 2):
        try:
            cur = _connection().cursor()
            try:
                cur.execute(sql, params)
                cols = [d[0].lower() for d in cur.description]
                rows = cur.fetchall()     # plain fetch: no pyarrow dependency (blocked on some hosts)
            finally:
                cur.close()
            df = pd.DataFrame(rows, columns=cols)
            return _coerce(df)
        except Exception as exc:  # noqa: BLE001
            if attempt == 1 and getattr(exc, "errno", None) == _TOKEN_EXPIRED:
                _connection.clear()
                continue
            raise
    raise RuntimeError("unreachable")


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Decimal -> float, date-like object columns -> datetime64 (fetchall returns Python objects)."""
    import datetime as _dt
    from decimal import Decimal

    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        first = s.iloc[0]
        if isinstance(first, Decimal):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        elif isinstance(first, (_dt.date, _dt.datetime)) and not isinstance(first, pd.Timestamp):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def query_many(sqls: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Run independent queries in parallel (cold marts take seconds each; parallel ≈ the slowest one)."""
    _connection()  # open once before fanning out
    with ThreadPoolExecutor(max_workers=min(6, len(sqls))) as pool:
        futures = {name: pool.submit(query, sql) for name, sql in sqls.items()}
        return {name: f.result() for name, f in futures.items()}
