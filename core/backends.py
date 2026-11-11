"""Django side of the platform backend switches (warehouse + LLM).

The switches themselves live in afya_backends.toggles; this module

* registers System Settings as the source of truth for Django processes
  (core/apps.py calls :func:`register` at startup),
* flips a switch for a superuser (:func:`switch`): saves the setting,
  audits it, then propagates it to the systems that can't read Django's
  database - Airflow (Variables) and Redash (moves saved queries between
  the Snowflake data source and its Databend twin) - and reports how each
  step went,
* reports health (:func:`status`): is each backend reachable, is Airflow
  in sync, where Redash queries currently point.

Propagation runs in a background thread so the settings page answers
immediately; its progress is kept in the cache and polled by the page.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from django.conf import settings
from django.core.cache import cache

from afya_backends import toggles

logger = logging.getLogger(__name__)

SETTING_KEYS = {toggles.WAREHOUSE: "warehouse_backend", toggles.LLM: "llm_provider"}
DESCRIPTIONS = {
    toggles.WAREHOUSE: "Data warehouse every pipeline, dashboard, agent and Redash query uses: "
                       "'snowflake' or 'databend'. Change it with the switch on this page.",
    toggles.LLM: "LLM provider for AI features: 'api' (hosted APIs via GROQ/XAI/ANTHROPIC/OPENAI keys) "
                 "or 'ollama' (local models). Change it with the switch on this page.",
}
PROGRESS_KEY = "afya_backend_switch_progress"
REDASH_MAP_KEY = "redash_warehouse_map"


# ---------------------------------------------------------------------------
# Source registration
# ---------------------------------------------------------------------------

def _settings_source(switch: str):
    key = SETTING_KEYS.get(switch)
    if not key:
        return None

    def read():
        from django.db import close_old_connections
        from .models import SystemSettings
        try:
            return SystemSettings.get(key)
        finally:
            close_old_connections()
    # Async callers (Channels consumers) may not touch the ORM on the event
    # loop thread; do the (cached, rare) lookup on a worker thread instead
    # of letting it fail and silently fall back to the default.
    try:
        import asyncio
        asyncio.get_running_loop()
    except RuntimeError:
        from .models import SystemSettings
        return SystemSettings.get(key)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(read).result(timeout=10)


def register() -> None:
    toggles.register_source(_settings_source)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def _timed(fn):
    t = time.monotonic()
    try:
        detail = fn()
        return {"ok": True, "detail": detail, "ms": int((time.monotonic() - t) * 1000)}
    except Exception as exc:  # noqa: BLE001 - health check reports everything
        return {"ok": False, "detail": str(exc)[:300], "ms": int((time.monotonic() - t) * 1000)}


def check_databend() -> dict:
    def go():
        from afya_backends.warehouse.dbapi import DatabendTransport
        t = DatabendTransport(timeout=15)
        try:
            v = t.run("SELECT version()").rows[0][0]
            dbs = t.run("SELECT COUNT(*) FROM system.databases WHERE name NOT IN "
                        "('system','information_schema','default','AFYA_SF_COMPAT')").rows[0][0]
            return f"{v.split('(')[0].strip()} · {dbs} schema(s)"
        finally:
            t.close()
    return _timed(go)


def check_snowflake() -> dict:
    def go():
        import snowflake.connector
        from .warehouse_creds import snowflake_kwargs
        kw = snowflake_kwargs()
        kw.update(login_timeout=12, network_timeout=15)
        conn = snowflake.connector.connect(**kw)
        try:
            cur = conn.cursor()
            cur.execute("SELECT CURRENT_VERSION()")
            return f"Snowflake {cur.fetchone()[0]}"
        finally:
            conn.close()
    return _timed(go)


def check_ollama() -> dict:
    def go():
        import requests
        from afya_backends import llm
        r = requests.get(llm.ollama_host() + "/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        want = llm.ollama_model()
        have = any(m == want or m.split(":")[0] == want.split(":")[0] for m in models)
        if not have:
            raise RuntimeError(f"model {want!r} not pulled (have: {', '.join(models) or 'none'})")
        return f"{want} ready · {len(models)} model(s)"
    return _timed(go)


def check_llm_api() -> dict:
    def go():
        from afya_backends import llm
        keys = llm.configured_api_providers()
        if not keys:
            raise RuntimeError("no GROQ_API_KEY / XAI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY set")
        chat_p = llm.api_chat_provider()
        chat_m = llm.groq_model() if chat_p == "groq" else llm.openai_fallback_model()
        emb = "local" if llm.embed_provider() == llm.EMBED_LOCAL else "openai"
        return ("keys: " + ", ".join(keys) + f" · chat → {chat_p} ({chat_m})"
                + f" · embeddings → {emb} ({llm.local_embed_model() if emb == 'local' else 'text-embedding-3-small'})")
    return _timed(go)


def airflow_variables() -> dict:
    def go():
        from airflow_ui.client import get_client
        c = get_client()
        out = {}
        for switch, var in toggles.AIRFLOW_VARIABLES.items():
            resp = c.request("GET", f"/variables/{var}")
            out[switch] = resp.json().get("value") if resp.ok else None
        return out
    return _timed(go)


def status(deep: bool = True) -> dict[str, Any]:
    toggles.clear_cache()
    wh, llm_p = toggles.warehouse_backend(), toggles.llm_provider()
    data: dict[str, Any] = {
        "warehouse_backend": wh,
        "llm_provider": llm_p,
        "progress": cache.get(PROGRESS_KEY),
        "redash": _redash_summary(),
    }
    if deep:
        checks = {}
        threads = []

        def run(name, fn):
            checks[name] = fn()
        for name, fn in (("databend", check_databend), ("snowflake", check_snowflake),
                         ("ollama", check_ollama), ("llm_api", check_llm_api), ("airflow", airflow_variables)):
            th = threading.Thread(target=run, args=(name, fn), daemon=True)
            th.start()
            threads.append(th)
        for th in threads:
            th.join(timeout=25)
        data["checks"] = checks
        af = checks.get("airflow") or {}
        if af.get("ok"):
            vals = af["detail"]
            data["airflow_in_sync"] = (vals.get(toggles.WAREHOUSE) == wh and vals.get(toggles.LLM) == llm_p)
    return data


def _redash_summary() -> dict:
    from .models import SystemSettings
    mapping = SystemSettings.get(REDASH_MAP_KEY) or {}
    return {"twins": len(mapping.get("twins", {})), "moved_queries": mapping.get("moved_count", 0),
            "pointing_at": mapping.get("pointing_at", "snowflake")}


# ---------------------------------------------------------------------------
# Switching
# ---------------------------------------------------------------------------

def _progress(steps: list, done: bool = False, **extra) -> None:
    cache.set(PROGRESS_KEY, {"steps": steps, "done": done, "at": time.time(), **extra}, 3600)


def switch(switch_name: str, value: str, user=None, propagate_async: bool = True) -> dict:
    """Flip ``switch_name`` to ``value`` (validated), save, audit, propagate."""
    from .models import AuditLog, Notification, SystemSettings

    if switch_name not in SETTING_KEYS:
        raise ValueError(f"unknown switch {switch_name!r}")
    norm = toggles.normalize(switch_name, value)
    if not norm:
        raise ValueError(f"{value!r} is not a valid value for {switch_name} "
                         f"(choose {', '.join(toggles.CHOICES[switch_name])})")
    previous = toggles.current(switch_name)
    SystemSettings.set(SETTING_KEYS[switch_name], norm, user=user, description=DESCRIPTIONS[switch_name])
    toggles.clear_cache()
    try:
        from django.contrib.auth import get_user_model
        AuditLog.log(user, "update", "PlatformBackend", resource_id=switch_name,
                     detail=f"{switch_name}: {previous} -> {norm}")
        for admin in get_user_model().objects.filter(is_superuser=True, is_active=True):
            if user is not None and admin.pk == getattr(user, "pk", None):
                continue
            Notification.send(admin, f"{'Warehouse' if switch_name == toggles.WAREHOUSE else 'LLM provider'} "
                                     f"switched to {norm}",
                              f"{getattr(user, 'username', 'system')} switched {switch_name} from {previous} to "
                              f"{norm}.", notification_type="warning", link="/core/settings/")
    except Exception:  # never let bookkeeping block the switch
        logger.exception("backend switch bookkeeping failed")
    logger.warning("Platform %s switched %s -> %s by %s", switch_name, previous, norm,
                   getattr(user, "username", "system"))
    steps = [{"name": "Saved setting", "ok": True, "detail": f"{previous} → {norm}"}]
    _progress(steps, switch=switch_name, value=norm)
    if propagate_async:
        threading.Thread(target=_propagate, args=(switch_name, norm, steps), daemon=True).start()
    else:
        _propagate(switch_name, norm, steps)
    return {"switch": switch_name, "previous": previous, "value": norm, "steps": steps}


def _propagate(switch_name: str, value: str, steps: list) -> None:
    from django.db import close_old_connections
    try:
        steps.append(_step("Airflow Variable", lambda: _sync_airflow(switch_name, value)))
        _progress(steps, switch=switch_name, value=value)
        if switch_name == toggles.WAREHOUSE:
            steps.append(_step("Redash queries", lambda: sync_redash(value)))
            _progress(steps, switch=switch_name, value=value)
            steps.append(_step("Cube", lambda: _sync_cube(value)))
    finally:
        _progress(steps, done=True, switch=switch_name, value=value)
        close_old_connections()


def _step(name, fn) -> dict:
    try:
        return {"name": name, "ok": True, "detail": fn() or "done"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("backend switch step %s failed", name)
        return {"name": name, "ok": False, "detail": str(exc)[:400]}


def _sync_airflow(switch_name: str, value: str) -> str:
    from airflow_ui.client import get_client
    c = get_client()
    var = toggles.AIRFLOW_VARIABLES[switch_name]
    body = {"key": var, "value": value,
            "description": f"Set by the dashboard's System Settings switch ({switch_name}). "
                           f"Pipelines read it at connect time; a DAG run's warehouse_backend param overrides it."}
    resp = c.request("PATCH", f"/variables/{var}", json=body)
    if resp.status_code == 404:
        resp = c.request("POST", "/variables", json=body)
    if not resp.ok:
        raise RuntimeError(f"Airflow returned {resp.status_code}: {resp.content[:200]!r}")
    return f"{var} = {value}"


def sync_all_to_airflow() -> list:
    return [_step(f"Airflow {s}", lambda s=s: _sync_airflow(s, toggles.current(s))) for s in SETTING_KEYS]


def _sync_cube(value: str) -> str:
    # cube/cube.js polls Django's backend status endpoint and keeps one
    # compiled app per warehouse (contextToAppId), so nothing to push.
    return "Cube follows the switch on its next query (polls every 15 s)"


# ---------------------------------------------------------------------------
# Redash
# ---------------------------------------------------------------------------

DATABEND_RUNNER_TYPE = "afya_databend"


def _redash():
    from redash_ui.client import get_client
    return get_client()


def _databend_ds_options() -> dict:
    return {
        "url": getattr(settings, "REDASH_DATABEND_URL", "") or "http://databend:8000",
        "user": settings.DATABEND_USER,
        "password": settings.DATABEND_PASSWORD,
        "translate_url": getattr(settings, "REDASH_TRANSLATE_URL", "") or "http://web:8000/core/api/warehouse/translate/",
        "translate_token": getattr(settings, "AFYA_INTERNAL_API_TOKEN", ""),
        "database": "HOSPITALS",
    }


def sync_redash(value: str) -> str:
    """Point every saved Redash query at the chosen warehouse.

    Each Snowflake data source gets a Databend twin (created on first use);
    switching moves queries between a source and its twin, remembering the
    pairing so switching back restores every query - including ones
    created while on Databend - to its Snowflake source.
    """
    from .models import SystemSettings
    rd = _redash()
    sources = rd.get_json("/api/data_sources")
    mapping = SystemSettings.get(REDASH_MAP_KEY) or {}
    twins: dict = mapping.get("twins", {})          # snowflake ds id -> databend ds id
    by_id = {str(s["id"]): s for s in sources}
    for s in sources:
        if s["type"] != "snowflake" or str(s["id"]) in twins and str(twins[str(s["id"])]) in by_id:
            continue
        payload = {"name": f"{s['name']} [databend]", "type": DATABEND_RUNNER_TYPE,
                   "options": _databend_ds_options()}
        resp = rd.request("POST", "/api/data_sources", json=payload)
        if not resp.ok:
            raise RuntimeError(f"could not create Databend twin for {s['name']!r}: {resp.status_code} "
                               f"{resp.content[:200]!r} - is the afya_databend runner installed in Redash "
                               f"(REDASH_ADDITIONAL_QUERY_RUNNERS)?")
        twins[str(s["id"])] = resp.json()["id"]
    reverse = {str(v): k for k, v in twins.items()}
    src_ids = set(twins) if value == toggles.DATABEND else set(reverse)
    target = (lambda ds: twins[ds]) if value == toggles.DATABEND else (lambda ds: int(reverse[ds]))
    moved, failed, page = 0, 0, 1
    while True:
        data = rd.get_json("/api/queries", page=page, page_size=100)
        for qry in data.get("results", []):
            ds = str(qry.get("data_source_id"))
            if ds not in src_ids:
                continue
            resp = rd.request("POST", f"/api/queries/{qry['id']}",
                              json={"data_source_id": int(target(ds)), "version": qry.get("version")})
            if resp.ok:
                moved += 1
            else:
                failed += 1
                logger.warning("Redash query %s not moved: %s", qry["id"], resp.status_code)
        if page * 100 >= data.get("count", 0):
            break
        page += 1
    SystemSettings.set(REDASH_MAP_KEY, {"twins": twins, "pointing_at": value,
                                        "moved_count": moved, "at": time.time()},
                       description="Pairs each Redash Snowflake data source with its Databend twin; "
                                   "maintained by the warehouse switch.")
    msg = f"{moved} quer{'y' if moved == 1 else 'ies'} now on {value}"
    if failed:
        raise RuntimeError(msg + f"; {failed} could not be moved (see logs)")
    return msg
