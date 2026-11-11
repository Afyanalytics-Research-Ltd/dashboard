"""One LLM entry point for hosted APIs or a local Ollama server.

The platform switch ``llm_provider`` (System Settings, Airflow Variable
``afya_llm_provider``, env ``AFYA_LLM_PROVIDER``; see toggles.py) picks:

* ``api``    - the hosted providers the features already use (OpenAI,
               Groq, xAI, Anthropic) with their API keys,
* ``ollama`` - a local Ollama server (OLLAMA_HOST, default
               http://localhost:11434) through its OpenAI-compatible
               /v1 API, so every OpenAI-SDK / LangChain call site works
               unchanged; model names are mapped to OLLAMA_MODEL.

Call sites ask this module for a client instead of constructing one:

    from afya_backends import llm
    client = llm.openai_client()                 # openai.OpenAI (or Ollama-backed)
    model = llm.model("gpt-4o-mini")             # same name on api, OLLAMA_MODEL on ollama
    text = llm.chat([{"role": "user", "content": "hi"}], max_tokens=200)   # any provider
    chat_model = llm.langchain_chat(temperature=0.3)                       # LangChain

Embeddings: hosted -> text-embedding-3-small; Ollama -> OLLAMA_EMBED_MODEL
(default nomic-embed-text).  Indexes built with one can't be queried with
the other (different vector sizes), so :func:`embeddings_index_path`
gives each provider its own index file.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterable, Optional

from . import toggles

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def provider() -> str:
    return toggles.llm_provider()


def is_ollama() -> bool:
    return provider() == toggles.LLM_OLLAMA


def ollama_host() -> str:
    host = (os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip()
    if not host.startswith("http"):
        host = "http://" + host
    return host.rstrip("/").removesuffix("/v1")


def ollama_model(purpose: Optional[str] = None) -> str:
    if purpose:
        specific = os.getenv(f"OLLAMA_MODEL_{purpose.upper()}", "").strip()
        if specific:
            return specific
    return (os.getenv("OLLAMA_MODEL") or "llama3.1").strip()


def ollama_embed_model() -> str:
    return (os.getenv("OLLAMA_EMBED_MODEL") or "nomic-embed-text").strip()


def _secret(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        try:
            from django.conf import settings
            v = str(getattr(settings, name, "") or "").strip()
        except Exception:
            v = ""
    if v in ("sk-...", "your-key", "changeme"):
        return ""
    return v


def configured_api_providers() -> list[str]:
    out = []
    for name, keys in (("openai", ("OPENAI_API_KEY",)), ("groq", ("GROQ_API_KEY", "GROQ_API")),
                       ("xai", ("XAI_API_KEY",)), ("anthropic", ("ANTHROPIC_API_KEY", "ANTHROPIC_API"))):
        if any(_secret(k) for k in keys):
            out.append(name)
    return out


def model(requested: Optional[str] = None, purpose: Optional[str] = None) -> str:
    """The model name to send: ``requested`` on hosted APIs, the Ollama
    model when local."""
    if is_ollama():
        return ollama_model(purpose)
    return requested or "gpt-4o-mini"


# ---------------------------------------------------------------------------
# clients
# ---------------------------------------------------------------------------

def openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
    """An ``openai.OpenAI`` client for the active provider.

    On ``api`` this is exactly what the call site built before (its own
    key / base_url).  On ``ollama`` it points at Ollama's OpenAI API.
    """
    from openai import OpenAI
    if is_ollama():
        return OpenAI(base_url=ollama_host() + "/v1", api_key="ollama",
                      timeout=kwargs.pop("timeout", 300), **kwargs)
    return OpenAI(api_key=api_key or _secret("OPENAI_API_KEY") or None, base_url=base_url, **kwargs)


def async_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
    from openai import AsyncOpenAI
    if is_ollama():
        return AsyncOpenAI(base_url=ollama_host() + "/v1", api_key="ollama",
                           timeout=kwargs.pop("timeout", 300), **kwargs)
    return AsyncOpenAI(api_key=api_key or _secret("OPENAI_API_KEY") or None, base_url=base_url, **kwargs)


def langchain_chat(temperature: float = 0.3, model_name: Optional[str] = None, api_key: Optional[str] = None,
                   **kwargs):
    """A LangChain chat model (tool-calling capable) for the active provider."""
    from langchain_openai import ChatOpenAI
    if is_ollama():
        return ChatOpenAI(base_url=ollama_host() + "/v1", api_key="ollama", model=ollama_model("agent"),
                          temperature=temperature, timeout=300, **kwargs)
    if model_name or api_key:
        return ChatOpenAI(model=model_name or "gpt-4o-mini", api_key=api_key or _secret("OPENAI_API_KEY"),
                          temperature=temperature, **kwargs)
    # Hosted priority order the agents already used: Groq -> xAI -> Anthropic -> OpenAI.
    if _secret("GROQ_API_KEY"):
        return ChatOpenAI(api_key=_secret("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1",
                          model="llama-3.3-70b-versatile", temperature=temperature, **kwargs)
    if _secret("XAI_API_KEY"):
        return ChatOpenAI(api_key=_secret("XAI_API_KEY"), base_url="https://api.x.ai/v1",
                          model="grok-3-mini", temperature=temperature, **kwargs)
    if _secret("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(api_key=_secret("ANTHROPIC_API_KEY"), model="claude-haiku-4-5",
                             temperature=temperature, **kwargs)
    if _secret("OPENAI_API_KEY"):
        return ChatOpenAI(api_key=_secret("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=temperature, **kwargs)
    raise RuntimeError("No LLM configured: set GROQ_API_KEY / XAI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY, "
                       "or switch System Settings → LLM provider to Ollama.")


# ---------------------------------------------------------------------------
# one-call chat helper (raw-HTTP / Anthropic SDK call sites)
# ---------------------------------------------------------------------------

def chat(messages: list[dict], *, system: Optional[str] = None, max_tokens: int = 1024,
         temperature: float = 0.2, prefer: Optional[Iterable[str]] = None, json_mode: bool = False,
         purpose: Optional[str] = None, timeout: float = 120) -> str:
    """Send a chat and return the reply text, on whichever provider is active.

    ``prefer`` orders hosted providers for ``api`` mode (default: the
    caller's existing order, else anthropic, groq, openai, xai).
    """
    msgs = list(messages)
    if system:
        msgs = [{"role": "system", "content": system}] + [m for m in msgs if m.get("role") != "system"]
    if is_ollama():
        client = openai_client()
        kw: dict[str, Any] = dict(model=ollama_model(purpose), messages=msgs, max_tokens=max_tokens,
                                  temperature=temperature)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        r = client.chat.completions.create(**kw)
        return r.choices[0].message.content or ""
    order = list(prefer or ("anthropic", "groq", "openai", "xai"))
    available = configured_api_providers()
    for p in order:
        if p not in available:
            continue
        if p == "anthropic":
            import anthropic
            sys_txt = "\n\n".join(m["content"] for m in msgs if m["role"] == "system") or None
            rest = [m for m in msgs if m["role"] != "system"]
            c = anthropic.Anthropic(api_key=_secret("ANTHROPIC_API_KEY") or _secret("ANTHROPIC_API"), timeout=timeout)
            kw = dict(model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"), max_tokens=max_tokens,
                      temperature=temperature, messages=rest)
            if sys_txt:
                kw["system"] = sys_txt
            r = c.messages.create(**kw)
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        base, key, mdl = {
            "groq": ("https://api.groq.com/openai/v1", _secret("GROQ_API_KEY") or _secret("GROQ_API"),
                     "llama-3.3-70b-versatile"),
            "xai": ("https://api.x.ai/v1", _secret("XAI_API_KEY"), "grok-3-mini"),
            "openai": (None, _secret("OPENAI_API_KEY"), "gpt-4o-mini"),
        }[p]
        from openai import OpenAI
        c = OpenAI(api_key=key, base_url=base, timeout=timeout)
        kw = dict(model=mdl, messages=msgs, max_tokens=max_tokens, temperature=temperature)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        r = c.chat.completions.create(**kw)
        return r.choices[0].message.content or ""
    raise RuntimeError("No hosted LLM API key configured (and the LLM switch is on 'api').")


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------

def embedding_model() -> str:
    return ollama_embed_model() if is_ollama() else "text-embedding-3-small"


def embed(texts: list[str], client=None) -> list[list[float]]:
    client = client or openai_client()
    r = client.embeddings.create(model=embedding_model(), input=texts)
    return [d.embedding for d in r.data]


def embeddings_index_path(default_path):
    """Hosted embeddings keep ``default_path``; Ollama uses a sibling file
    (embeddings.npz -> embeddings.ollama.npz) built with its own model."""
    from pathlib import Path
    if not is_ollama():
        return default_path
    p = Path(default_path)
    out = p.with_name(p.stem + ".ollama" + p.suffix)
    return out if isinstance(default_path, Path) else str(out)


def describe() -> dict:
    """What the switch currently resolves to (for status pages / logs)."""
    if is_ollama():
        return {"provider": "ollama", "host": ollama_host(), "model": ollama_model(),
                "embed_model": ollama_embed_model()}
    return {"provider": "api", "configured": configured_api_providers()}


def ollama_json(prompt: str, schema_hint: str = "") -> Any:
    """Small convenience: ask for JSON and parse it (tolerates code fences)."""
    raw = chat([{"role": "user", "content": prompt + ("\n\n" + schema_hint if schema_hint else "")}],
               json_mode=True)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)
