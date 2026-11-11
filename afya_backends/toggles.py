"""Which warehouse / which LLM provider is active right now.

Two platform-wide switches, flipped by a superuser on System Settings
(core.views.SystemSettingsView, stored in core.SystemSettings):

    warehouse_backend   "snowflake" | "databend"
    llm_provider        "api"       | "ollama"

The same answer must be reachable from three very different processes:

* Django (web, Celery) reads core.SystemSettings directly - Django
  registers that reader at startup (core/apps.py -> register_source()).
* Airflow tasks can't see Django's database.  When the switch is flipped,
  Django also writes Airflow Variables ``afya_warehouse_backend`` /
  ``afya_llm_provider`` (afya_backends.sync), and this module reads them
  inside a task.  A DAG run can override the warehouse with its
  ``warehouse_backend`` trigger param (config/airflow_local_settings.py).
* Anything else (CLI scripts, tests) uses environment variables.

Resolution order for each switch (first hit wins):

    1. AFYA_<SWITCH>  environment variable   - explicit override: a DAG
       run's param sets this for its task process; tests and one-off CLI
       runs set it by hand.
    2. registered sources (Django SystemSettings, Airflow Variable)
    3. WAREHOUSE_BACKEND / LLM_PROVIDER_DEFAULT env (deploy-time default)
    4. built-in default ("snowflake", "api")

Answers from (2) are cached for a few seconds so a busy dashboard doesn't
hit the settings table for every query, while a flip still takes effect
almost immediately everywhere.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

WAREHOUSE = "warehouse_backend"
LLM = "llm_provider"

SNOWFLAKE = "snowflake"
DATABEND = "databend"
LLM_API = "api"
LLM_OLLAMA = "ollama"

CHOICES = {
    WAREHOUSE: (SNOWFLAKE, DATABEND),
    LLM: (LLM_API, LLM_OLLAMA),
}
DEFAULTS = {WAREHOUSE: SNOWFLAKE, LLM: LLM_API}
# Deploy-time defaults (used until a superuser flips the switch).  Not plain
# LLM_PROVIDER: pipelines/competitor_pricing_agent.py historically used that
# name with different values (ollama|anthropic).
DEPLOY_DEFAULT_ENV = {WAREHOUSE: ("WAREHOUSE_BACKEND",), LLM: ("LLM_PROVIDER_DEFAULT",)}
# Airflow Variable names written by afya_backends.sync.
AIRFLOW_VARIABLES = {WAREHOUSE: "afya_warehouse_backend", LLM: "afya_llm_provider"}

CACHE_SECONDS = float(os.getenv("AFYA_TOGGLE_CACHE_SECONDS", "5"))

_sources: list[Callable[[str], Optional[str]]] = []
_cache: dict[str, tuple[float, str]] = {}
_lock = threading.Lock()
_local = threading.local()


def normalize(switch: str, value) -> Optional[str]:
    """Return the canonical value or None if ``value`` isn't valid."""
    if value is None:
        return None
    if isinstance(value, dict):  # SystemSettings may hold {"value": ...}
        value = value.get("value")
    v = str(value).strip().lower().strip('"')
    aliases = {
        "sf": SNOWFLAKE, "snow": SNOWFLAKE,
        "db": DATABEND, "datablend": DATABEND, "data_bend": DATABEND,
        "cloud": LLM_API, "apikeys": LLM_API, "api_keys": LLM_API, "keys": LLM_API, "llm_api": LLM_API,
        "local": LLM_OLLAMA,
    }
    v = aliases.get(v, v)
    return v if v in CHOICES[switch] else None


def register_source(fn: Callable[[str], Optional[str]]) -> None:
    """Add a lookup ``fn(switch) -> value|None`` (Django / Airflow)."""
    if fn not in _sources:
        _sources.append(fn)
    clear_cache()


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def _env_override(switch: str) -> Optional[str]:
    return normalize(switch, os.getenv("AFYA_" + switch.upper()))


def current(switch: str) -> str:
    """The active value of ``switch`` (see module docstring for order)."""
    forced = getattr(_local, "overrides", {}).get(switch)
    if forced:
        return forced
    v = _env_override(switch)
    if v:
        return v
    now = time.monotonic()
    with _lock:
        hit = _cache.get(switch)
        if hit and now - hit[0] < CACHE_SECONDS:
            return hit[1]
    value = None
    for fn in list(_sources) + [_airflow_variable_source]:
        try:
            value = normalize(switch, fn(switch))
        except Exception as exc:  # a broken source must never break queries
            logger.debug("toggle source %r failed for %s: %s", fn, switch, exc)
            value = None
        if value:
            break
    if not value:
        value = next((v for v in (normalize(switch, os.getenv(name)) for name in DEPLOY_DEFAULT_ENV[switch]) if v),
                     DEFAULTS[switch])
    with _lock:
        _cache[switch] = (now, value)
    return value


def warehouse_backend() -> str:
    return current(WAREHOUSE)


def llm_provider() -> str:
    return current(LLM)


def is_databend() -> bool:
    return warehouse_backend() == DATABEND


def state_path(path):
    """Per-warehouse location of a pipeline progress file.

    Watermarks/page checkpoints record what has been loaded INTO the
    warehouse, so each warehouse needs its own: on Databend ``.watermarks.json``
    becomes ``.watermarks.databend.json`` (a fresh file => full backfill from
    the source APIs), while Snowflake keeps using the original file
    untouched - switching back resumes exactly where Snowflake stopped.
    Accepts str or Path; returns the same type.
    """
    from pathlib import Path
    if warehouse_backend() != DATABEND:
        return path
    p = Path(path)
    name = p.name
    if p.suffix:
        new = name[: -len(p.suffix)] + ".databend" + p.suffix
    else:
        new = name + ".databend"
    out = p.with_name(new)
    return out if isinstance(path, Path) else str(out)


def state_key(key: str) -> str:
    """Per-warehouse name for a watermark kept in an Airflow Variable or a
    state table key (see :func:`state_path`)."""
    return key if warehouse_backend() != DATABEND else f"{key}__databend"


class override:
    """Temporarily force a switch in this thread (tests, copy tools)::

        with toggles.override(WAREHOUSE, DATABEND):
            ...
    """

    def __init__(self, switch: str, value: str):
        v = normalize(switch, value)
        if not v:
            raise ValueError(f"{value!r} is not a valid {switch}")
        self.switch, self.value = switch, v

    def __enter__(self):
        ov = getattr(_local, "overrides", None)
        if ov is None:
            ov = _local.overrides = {}
        self.prev = ov.get(self.switch)
        ov[self.switch] = self.value
        return self.value

    def __exit__(self, *exc):
        if self.prev is None:
            _local.overrides.pop(self.switch, None)
        else:
            _local.overrides[self.switch] = self.prev
        return False


def _airflow_variable_source(switch: str) -> Optional[str]:
    """Inside an Airflow task: the run's trigger param, then the Variable."""
    if "airflow" not in __import__("sys").modules:
        return None
    try:
        from airflow.sdk import get_current_context
        ctx = get_current_context()
        params = ctx.get("params") or {}
        if switch == WAREHOUSE:
            p = normalize(WAREHOUSE, params.get("warehouse_backend"))
            if p:
                return p
    except Exception:
        pass
    try:
        from airflow.sdk import Variable
        return Variable.get(AIRFLOW_VARIABLES[switch], default=None)
    except Exception:
        try:
            from airflow.models import Variable as ModelVariable  # Airflow 2 / api-server side
            return ModelVariable.get(AIRFLOW_VARIABLES[switch], default_var=None)
        except Exception:
            return None
