"""Browserbase (browserbase.com) session client for the browser-automation
console.

Browserbase is a hosted headless-browser SaaS — there is nothing to
self-host. The SDK talks to api.browserbase.com directly, and the
live-view iframe src returned by ``get_live_urls`` points at Browserbase's
own domain, loaded straight in the viewer's browser rather than proxied
through this app.
"""

import logging
from typing import Optional

from browserbase import Browserbase
from django.conf import settings

logger = logging.getLogger(__name__)


class BrowserbaseError(Exception):
    """Raised when a Browserbase API call fails or is misconfigured."""


def _client() -> Browserbase:
    if not settings.BROWSERBASE_API_KEY:
        raise BrowserbaseError(
            "BROWSERBASE_API_KEY is not configured — set it in .env."
        )
    return Browserbase(api_key=settings.BROWSERBASE_API_KEY)


def _require_project_id() -> str:
    if not settings.BROWSERBASE_PROJECT_ID:
        raise BrowserbaseError(
            "BROWSERBASE_PROJECT_ID is not configured — set it in .env."
        )
    return settings.BROWSERBASE_PROJECT_ID


def list_sessions(status: Optional[str] = None) -> list[dict]:
    """Return sessions, most-recently-created first.

    ``status`` filters to one of PENDING/RUNNING/ERROR/TIMED_OUT/COMPLETED.
    """
    client = _client()
    try:
        kwargs = {"status": status} if status else {}
        sessions = client.sessions.list(**kwargs)
    except Exception as exc:
        logger.error("Browserbase list_sessions failed: %s", exc)
        raise BrowserbaseError(str(exc)) from exc
    return sorted(
        (s.model_dump() for s in sessions),
        key=lambda s: s.get("created_at") or "",
        reverse=True,
    )


def create_session(
    region: Optional[str] = None,
    keep_alive: bool = False,
    timeout: Optional[int] = None,
    use_proxy: bool = False,
) -> dict:
    """Start a new session and return it (id, status, region, ...).

    Always attaches the persistent Browserbase Context (BROWSERBASE_CONTEXT_ID)
    when one is configured, so a login made in this session's live view
    carries over into every other session — console or chat — that shares
    the same context.

    ``use_proxy=True`` routes the session's traffic through the Bright Data
    Kenya proxy (BRIGHTDATA_PROXY_*) instead of Browserbase's own network,
    so the outbound IP geolocates to Kenya. Browserbase's own geolocation-
    targeted managed proxies need a paid plan (confirmed via a live 402 on
    the free plan) — this is the free-plan-compatible alternative.
    """
    client = _client()
    project_id = _require_project_id()
    kwargs = {"project_id": project_id, "keep_alive": keep_alive}
    if region:
        kwargs["region"] = region
    if timeout:
        kwargs["timeout"] = timeout

    browser_settings = {}
    if settings.BROWSERBASE_CONTEXT_ID:
        browser_settings["context"] = {"id": settings.BROWSERBASE_CONTEXT_ID, "persist": True}
    if browser_settings:
        kwargs["browser_settings"] = browser_settings

    if use_proxy:
        if not settings.BRIGHTDATA_PROXY_SERVER:
            raise BrowserbaseError(
                "BRIGHTDATA_PROXY_SERVER is not configured — set it in .env."
            )
        kwargs["proxies"] = [{
            "type": "external",
            "server": settings.BRIGHTDATA_PROXY_SERVER,
            "username": settings.BRIGHTDATA_PROXY_USERNAME,
            "password": settings.BRIGHTDATA_PROXY_PASSWORD,
        }]

    try:
        session = client.sessions.create(**kwargs)
    except Exception as exc:
        logger.error("Browserbase create_session failed: %s", exc)
        raise BrowserbaseError(str(exc)) from exc
    return session.model_dump()


def get_session(session_id: str) -> dict:
    client = _client()
    try:
        session = client.sessions.retrieve(session_id)
    except Exception as exc:
        logger.error("Browserbase get_session(%s) failed: %s", session_id, exc)
        raise BrowserbaseError(str(exc)) from exc
    return session.model_dump()


def get_live_urls(session_id: str) -> dict:
    """Live/debug view URLs for an active session.

    The console's live-view iframe src comes from ``debugger_fullscreen_url``
    here — only populated while the session's status is RUNNING.
    """
    client = _client()
    try:
        live = client.sessions.debug(session_id)
    except Exception as exc:
        logger.error("Browserbase get_live_urls(%s) failed: %s", session_id, exc)
        raise BrowserbaseError(str(exc)) from exc
    return live.model_dump()


def end_session(session_id: str) -> dict:
    """Request Browserbase release the session (stops usage billing)."""
    client = _client()
    project_id = _require_project_id()
    try:
        session = client.sessions.update(
            session_id, status="REQUEST_RELEASE", project_id=project_id,
        )
    except Exception as exc:
        logger.error("Browserbase end_session(%s) failed: %s", session_id, exc)
        raise BrowserbaseError(str(exc)) from exc
    return session.model_dump()
