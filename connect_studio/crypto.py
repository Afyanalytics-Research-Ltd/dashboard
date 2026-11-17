"""Secret fields at rest: Fernet, keyed by CONNECT_STUDIO_KEY (or derived from SECRET_KEY)."""

from __future__ import annotations

import base64
import hashlib
import json
import os

from cryptography.fernet import Fernet, InvalidToken
from django.conf import settings


def _fernet() -> Fernet:
    raw = os.getenv("CONNECT_STUDIO_KEY") or f"connect-studio::{settings.SECRET_KEY}"
    key = base64.urlsafe_b64encode(hashlib.sha256(raw.encode()).digest())
    return Fernet(key)


def encrypt(data: dict) -> str:
    if not data:
        return ""
    return _fernet().encrypt(json.dumps(data).encode()).decode()


def decrypt(token: str) -> dict:
    if not token:
        return {}
    try:
        return json.loads(_fernet().decrypt(token.encode()).decode())
    except (InvalidToken, ValueError):
        return {}
