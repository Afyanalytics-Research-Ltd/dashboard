"""Groq (OpenAI-compatible) JSON calls for the mapper.

Uses the platform's Groq settings from ``afya_backends.llm`` (GROQ_API_KEY,
GROQ_MODEL — default the open-weight ``openai/gpt-oss-120b``). Every call is
JSON mode, retried on rate limits, and its token usage is accounted so the UI
can show what a mapping cost.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """No Groq key configured — the mapper falls back to deterministic matching."""


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    failures: int = 0
    model: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, resp, seconds: float) -> None:
        u = getattr(resp, "usage", None)
        with self._lock:
            self.calls += 1
            self.seconds += seconds
            if u is not None:
                self.prompt_tokens += int(getattr(u, "prompt_tokens", 0) or 0)
                self.completion_tokens += int(getattr(u, "completion_tokens", 0) or 0)

    def as_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens, "seconds": round(self.seconds, 1),
                "failures": self.failures, "model": self.model}


def settings() -> dict:
    from afya_backends import llm as platform_llm

    key = platform_llm.groq_key()
    return {"available": bool(key), "model": platform_llm.groq_model("mapping"), "provider": "groq",
            "base_url": platform_llm.GROQ_BASE_URL}


def _client():
    from openai import OpenAI
    from afya_backends import llm as platform_llm

    key = platform_llm.groq_key()
    if not key:
        raise LLMUnavailable("GROQ_API_KEY is not set — AI mapping is off; only rule-based matches will be suggested.")
    return OpenAI(api_key=key, base_url=platform_llm.GROQ_BASE_URL, timeout=180, max_retries=0)


def parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _json_generation_failed(e: Exception) -> bool:
    """Groq 400 json_validate_failed — the openai SDK's str(e) shows only the message, the code is in .code/.body."""
    body = getattr(e, "body", None)
    code = getattr(e, "code", None) or (body.get("code") if isinstance(body, dict) else None)
    return code == "json_validate_failed" or "json_validate_failed" in str(e) or "Failed to generate JSON" in str(e)


def chat_json(system: str, user: str, *, usage: Usage | None = None, model: str | None = None,
              max_tokens: int = 8000, temperature: float = 0.1, retries: int = 7) -> dict:
    from afya_backends import llm as platform_llm

    client = _client()
    mdl = model or platform_llm.groq_model("mapping")
    if usage is not None:
        usage.model = mdl
    kw = dict(model=mdl, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
              temperature=temperature, max_tokens=max_tokens, response_format={"type": "json_object"})
    kw.update(platform_llm.groq_extra(mdl))
    if "reasoning_effort" in kw:
        # mapping quality beats latency here (CONNECT_STUDIO_REASONING_EFFORT=low|medium|high)
        kw["reasoning_effort"] = os.getenv("CONNECT_STUDIO_REASONING_EFFORT", "medium")
    delay = 3.0
    last: Exception | None = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(**kw)
            if usage is not None:
                usage.add(resp, time.time() - t0)
            content = resp.choices[0].message.content or ""
            return parse_json(content)
        except json.JSONDecodeError as e:
            last = e
            kw["temperature"] = 0.0
            logger.warning("Groq returned non-JSON (attempt %d)", attempt + 1)
        except Exception as e:  # rate limits / transient errors
            last = e
            status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            if status in (400, 401, 403, 404) and not _json_generation_failed(e):
                break
            if _json_generation_failed(e):
                # Groq's JSON mode occasionally rejects its own output — retry, deterministically
                kw["temperature"] = 0.0
            retry_after = None
            try:
                retry_after = float(e.response.headers.get("retry-after"))  # type: ignore[attr-defined]
            except Exception:
                pass
            # Groq says how long to wait: "Please try again in 7.66s" / "in 450ms" / "in 1m12s"
            m = re.search(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)", str(e))
            if m:
                secs = float(m.group(2)) / (1000 if m.group(3) == "ms" else 1) + 60 * int(m.group(1) or 0)
                retry_after = max(retry_after or 0, secs + 0.5)
            if retry_after and retry_after > 120:
                break  # a daily quota, not a burst limit — don't hang the job
            wait = retry_after or delay
            logger.warning("Groq call failed (%s); retry in %.0fs", str(e)[:120], wait)
            time.sleep(min(wait, 90))
            delay = min(delay * 2, 60)
    if usage is not None:
        with usage._lock:
            usage.failures += 1
    raise RuntimeError(f"Groq call failed: {str(last)[:300]}")
