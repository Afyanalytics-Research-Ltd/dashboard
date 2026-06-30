"""
Cube.dev API client.

Wraps the two endpoints the agent uses:
  - GET  /cubejs-api/v1/meta  → schema discovery
  - POST /cubejs-api/v1/load  → run a query

Auth: Bearer token (Cube API secret or signed JWT).
Set CUBE_API_URL and CUBE_API_TOKEN in your environment / Django settings.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from django.conf import settings

logger = logging.getLogger(__name__)


def _base_url() -> str:
    url = getattr(settings, "CUBE_API_URL", os.getenv("CUBE_API_URL", ""))
    if not url:
        raise RuntimeError("CUBE_API_URL is not configured.")
    return url.rstrip("/")


def _token() -> str:
    token = getattr(settings, "CUBE_API_TOKEN", os.getenv("CUBE_API_TOKEN", ""))
    if not token:
        raise RuntimeError("CUBE_API_TOKEN is not configured.")
    return token


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_token()}",
        "Content-Type": "application/json",
    }


def fetch_meta() -> dict[str, Any]:
    """
    Fetch the Cube schema (cubes, measures, dimensions).
    Used by the Schema Loader / Intent Classifier for grounding.
    """
    url = f"{_base_url()}/cubejs-api/v1/meta"
    with httpx.Client(timeout=15) as client:
        resp = client.get(url, headers=_headers())
        resp.raise_for_status()
        return resp.json()


def _clean_query(query: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitise a Cube query before sending:
      - Remove empty lists for filters / dimensions / timeDimensions
        (Cube rejects some empty-list edge cases depending on version).
      - Strip any None values from the top level.
    """
    cleaned = {}
    for key, value in query.items():
        if value is None:
            continue
        if isinstance(value, list) and len(value) == 0 and key in ("filters", "dimensions", "timeDimensions"):
            continue  # omit empty optional lists
        cleaned[key] = value
    return cleaned


def run_query(query: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a Cube query.

    Args:
        query: A valid Cube query dict
                {measures, dimensions, timeDimensions, filters, limit, …}

    Returns:
        The full Cube API response including data, annotation, and query.

    Raises:
        httpx.HTTPStatusError: on 4xx / 5xx responses
        httpx.TimeoutException: if Cube takes too long
    """
    url = f"{_base_url()}/cubejs-api/v1/load"
    clean = _clean_query(query)

    logger.info("run_query → %s", json.dumps(clean, indent=2))

    payload = {"query": clean}

    with httpx.Client(timeout=30) as client:
        resp = client.post(url, headers=_headers(), json=payload)
        if not resp.is_success:
            logger.error("Cube 400 body: %s", resp.text)
        resp.raise_for_status()
        return resp.json()