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

Hosted chat routing (``api`` mode): when GROQ_API_KEY is set, OpenAI-SDK
call sites (:func:`openai_client` / :func:`model`) run on Groq's hosted
open-source models (GROQ_MODEL, default openai/gpt-oss-120b) instead of
paid OpenAI, falling back to OpenAI (OPENAI_FALLBACK_MODEL, default
gpt-4o-mini) when Groq rate-limits or is unreachable.  Set
LLM_API_CHAT_PROVIDER=openai to send them straight to OpenAI again.
Embeddings (:func:`embed_provider`): ``api`` mode computes them in-process
with fastembed (LOCAL_EMBED_MODEL, default BAAI/bge-small-en-v1.5, free,
CPU) unless LLM_EMBED_PROVIDER=openai (text-embedding-3-small); Ollama mode
uses OLLAMA_EMBED_MODEL (default nomic-embed-text).  Indexes built with one
can't be queried with another (different vector sizes), so
:func:`embeddings_index_path` gives each provider its own index file.
"""

from __future__ import annotations

import json
import logging
import os
import re
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


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_OPENAI_MODEL_RE = re.compile(r"^(gpt-|chatgpt-|o[1-9](-|$))")


def groq_key() -> str:
    return _secret("GROQ_API_KEY") or _secret("GROQ_API")


def groq_model(purpose: Optional[str] = None) -> str:
    if purpose:
        specific = os.getenv(f"GROQ_MODEL_{purpose.upper()}", "").strip()
        if specific:
            return specific
    # Open-weight (Apache-2.0) model with tool calling + JSON mode on Groq.
    # llama-3.3-70b-versatile, the old default, was retired by Groq.
    return (os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b").strip()


def groq_extra(model_name: str) -> dict:
    """Extra request fields for Groq: gpt-oss models are reasoning models;
    low effort keeps latency down (GROQ_REASONING_EFFORT, blank = off)."""
    effort = os.getenv("GROQ_REASONING_EFFORT", "low").strip()
    if effort and "gpt-oss" in (model_name or ""):
        return {"reasoning_effort": effort}
    return {}


def openai_fallback_model() -> str:
    return (os.getenv("OPENAI_FALLBACK_MODEL") or "gpt-4o-mini").strip()


def api_chat_provider() -> str:
    """Where OpenAI-SDK chat calls go in ``api`` mode: ``groq`` (free/cheap
    open-source models) when a Groq key exists, else ``openai``.
    LLM_API_CHAT_PROVIDER=openai|groq overrides."""
    forced = os.getenv("LLM_API_CHAT_PROVIDER", "").strip().lower()
    if forced == "openai":
        return "openai"
    if forced in ("", "groq") and groq_key():
        return "groq"
    return "openai"


def _routes_to_groq() -> bool:
    return not is_ollama() and api_chat_provider() == "groq"


# ---------------------------------------------------------------------------
# embeddings provider
# ---------------------------------------------------------------------------

EMBED_LOCAL = "local"    # fastembed, in-process on CPU (free, open-source)
EMBED_OPENAI = "openai"  # text-embedding-3-small (paid)
EMBED_OLLAMA = "ollama"


def embed_provider() -> str:
    """Who computes embeddings: ``ollama`` when the LLM switch is on Ollama;
    otherwise ``local`` (fastembed, LOCAL_EMBED_MODEL) unless
    LLM_EMBED_PROVIDER=openai."""
    if is_ollama():
        return EMBED_OLLAMA
    forced = os.getenv("LLM_EMBED_PROVIDER", "").strip().lower()
    return EMBED_OPENAI if forced == EMBED_OPENAI else EMBED_LOCAL


def local_embed_model() -> str:
    return (os.getenv("LOCAL_EMBED_MODEL") or "BAAI/bge-small-en-v1.5").strip()


_local_models: dict[str, Any] = {}
_local_lock = __import__("threading").Lock()


def _local_text_embedding(name: str):
    """Load (once per process) a fastembed model.  Weights are cached under
    FASTEMBED_CACHE_PATH (the web image pre-downloads them there)."""
    m = _local_models.get(name)
    if m is None:
        with _local_lock:
            m = _local_models.get(name)
            if m is None:
                from fastembed import TextEmbedding
                cache = os.getenv("FASTEMBED_CACHE_PATH") or None
                m = TextEmbedding(name, cache_dir=cache) if cache else TextEmbedding(name)
                _local_models[name] = m
    return m


class _LocalEmbeddings:
    """``client.embeddings`` stand-in: ``create(model=, input=)`` answered
    in-process by fastembed, shaped like the OpenAI response."""

    def create(self, *, model: Optional[str] = None, input, **_ignored):
        from types import SimpleNamespace
        texts = [input] if isinstance(input, str) else list(input)
        name = model if model and "/" in model else local_embed_model()
        vecs = _local_text_embedding(name).embed(texts)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(x) for x in v], index=i)
                                     for i, v in enumerate(vecs)], model=name)


class _AsyncLocalEmbeddings(_LocalEmbeddings):
    async def create(self, **kw):
        import asyncio
        return await asyncio.to_thread(_LocalEmbeddings.create, self, **kw)


def local_embed(texts: list[str], *, query: bool = False, model_name: Optional[str] = None) -> list[list[float]]:
    """Embed in-process with fastembed.  ``query=True`` uses the model's
    query form (bge adds its retrieval instruction), for search queries
    matched against passages embedded with ``query=False``."""
    m = _local_text_embedding(model_name or local_embed_model())
    vecs = m.query_embed(texts) if query else m.embed(texts)
    return [[float(x) for x in v] for v in vecs]


def local_rerank_model() -> str:
    return (os.getenv("LOCAL_RERANK_MODEL") or "Xenova/ms-marco-MiniLM-L-6-v2").strip()


def local_rerank(query: str, documents: list[str], model_name: Optional[str] = None) -> list[float]:
    """Cross-encoder relevance scores (higher = more relevant), one per document."""
    if not documents:
        return []
    name = model_name or local_rerank_model()
    key = "rerank:" + name
    enc = _local_models.get(key)
    if enc is None:
        with _local_lock:
            enc = _local_models.get(key)
            if enc is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                cache = os.getenv("FASTEMBED_CACHE_PATH") or None
                enc = TextCrossEncoder(name, cache_dir=cache) if cache else TextCrossEncoder(name)
                _local_models[key] = enc
    return [float(s) for s in enc.rerank(query, documents)]


def model(requested: Optional[str] = None, purpose: Optional[str] = None) -> str:
    """The model name to send: the Ollama model when local; on hosted APIs
    ``requested``, except that OpenAI model names become the Groq model when
    chat is routed to Groq (see :func:`openai_client`)."""
    if is_ollama():
        return ollama_model(purpose)
    if _routes_to_groq() and (not requested or _OPENAI_MODEL_RE.match(requested)):
        return groq_model(purpose)
    return requested or "gpt-4o-mini"


# ---------------------------------------------------------------------------
# clients
# ---------------------------------------------------------------------------

def _fallback_worthy(exc: Exception) -> bool:
    """Groq errors worth retrying on OpenAI: rate limits / quota, outages,
    network trouble.  Bad requests (our bug) are re-raised as-is."""
    import openai
    return isinstance(exc, (openai.RateLimitError, openai.InternalServerError,
                            openai.APIConnectionError, openai.APITimeoutError))


def _openai_fallback_kwargs(kw: dict) -> dict:
    out = {k: v for k, v in kw.items() if k != "reasoning_effort"}
    out["model"] = openai_fallback_model()
    return out


class _GroqCompletions:
    def __init__(self, router):
        self._r = router

    @staticmethod
    def _groq_kwargs(kw: dict) -> dict:
        return {**groq_extra(kw.get("model", "")), **kw}

    def _fallback(self, e: Exception):
        fb = self._r._openai_or_none()
        if fb is None or not _fallback_worthy(e):
            raise e
        logger.warning("Groq chat failed (%s: %s); falling back to OpenAI %s",
                       type(e).__name__, e, openai_fallback_model())
        return fb

    def create(self, **kw):
        try:
            return self._r._groq.chat.completions.create(**self._groq_kwargs(kw))
        except Exception as e:
            return self._fallback(e).chat.completions.create(**_openai_fallback_kwargs(kw))


class _AsyncGroqCompletions(_GroqCompletions):
    async def create(self, **kw):
        try:
            return await self._r._groq.chat.completions.create(**self._groq_kwargs(kw))
        except Exception as e:
            return await self._fallback(e).chat.completions.create(**_openai_fallback_kwargs(kw))


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class _RoutedClient:
    """OpenAI-SDK-shaped client for ``api`` mode.  Chat -> Groq (OpenAI
    fallback) when ``groq_client`` is given, else OpenAI; embeddings ->
    in-process fastembed when ``local_embed``, else OpenAI; anything else ->
    OpenAI.  Covers the surface the call sites use (``chat.completions.create``,
    ``embeddings.create``)."""

    _completions_cls = _GroqCompletions
    _local_embeddings_cls = _LocalEmbeddings

    def __init__(self, groq_client, openai_factory, local_embed: bool = False):
        self._groq = groq_client
        self._openai_factory = openai_factory
        self._openai = None
        if groq_client is not None:
            self.chat = _Chat(self._completions_cls(self))
        if local_embed:
            self.embeddings = self._local_embeddings_cls()

    def _openai_or_none(self):
        if self._openai is None:
            try:
                self._openai = self._openai_factory()
            except Exception:  # no OPENAI_API_KEY
                return None
        return self._openai

    def __getattr__(self, name):  # chat / embeddings not routed above, models, ...
        if name.startswith("_"):
            raise AttributeError(name)
        c = self._openai_or_none()
        if c is None:
            raise RuntimeError(f"client.{name} needs OPENAI_API_KEY in 'api' mode")
        return getattr(c, name)


class _AsyncRoutedClient(_RoutedClient):
    _completions_cls = _AsyncGroqCompletions
    _local_embeddings_cls = _AsyncLocalEmbeddings


def _api_client(openai_cls, routed_cls, api_key, base_url, kwargs):
    make_openai = lambda: openai_cls(api_key=api_key or _secret("OPENAI_API_KEY") or None,  # noqa: E731
                                     base_url=base_url, **kwargs)
    if base_url is not None:  # explicit endpoint: the caller knows what it wants
        return make_openai()
    groq = None
    if api_chat_provider() == "groq":
        groq = openai_cls(api_key=groq_key(), base_url=GROQ_BASE_URL, max_retries=1,
                          **{k: v for k, v in kwargs.items() if k != "max_retries"})
    local_embed = embed_provider() == EMBED_LOCAL
    if groq is None and not local_embed:
        return make_openai()
    return routed_cls(groq, make_openai, local_embed=local_embed)


def openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
    """An ``openai.OpenAI``-compatible client for the active provider.

    On ``ollama`` it points at Ollama's OpenAI API.  On ``api``: chat goes
    to Groq when a Groq key exists (see :func:`api_chat_provider`; OpenAI is
    the fallback) and embeddings are computed locally by fastembed (see
    :func:`embed_provider`); pair it with :func:`model` /
    :func:`embedding_model`.  An explicit ``base_url`` bypasses the routing.
    """
    from openai import OpenAI
    if is_ollama():
        return OpenAI(base_url=ollama_host() + "/v1", api_key="ollama",
                      timeout=kwargs.pop("timeout", 300), **kwargs)
    return _api_client(OpenAI, _RoutedClient, api_key, base_url, kwargs)


def async_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
    from openai import AsyncOpenAI
    if is_ollama():
        return AsyncOpenAI(base_url=ollama_host() + "/v1", api_key="ollama",
                           timeout=kwargs.pop("timeout", 300), **kwargs)
    return _api_client(AsyncOpenAI, _AsyncRoutedClient, api_key, base_url, kwargs)


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
    if groq_key():
        m = groq_model("agent")
        return ChatOpenAI(api_key=groq_key(), base_url=GROQ_BASE_URL, model=m, temperature=temperature,
                          **{**groq_extra(m), **kwargs})
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
    candidates = [p for p in order if p in available]
    last_exc: Optional[Exception] = None
    for i, p in enumerate(candidates):
        try:
            return _chat_one(p, msgs, max_tokens=max_tokens, temperature=temperature,
                             json_mode=json_mode, purpose=purpose, timeout=timeout)
        except Exception as e:  # rate limit / outage on one provider -> try the next
            last_exc = e
            if i + 1 < len(candidates):
                logger.warning("LLM chat via %s failed (%s: %s); trying %s",
                               p, type(e).__name__, e, candidates[i + 1])
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No hosted LLM API key configured (and the LLM switch is on 'api').")


def _chat_one(p: str, msgs: list[dict], *, max_tokens: int, temperature: float, json_mode: bool,
              purpose: Optional[str], timeout: float) -> str:
    if p == "anthropic":
        import anthropic
        sys_txt = "\n\n".join(m["content"] for m in msgs if m["role"] == "system") or None
        rest = [m for m in msgs if m["role"] != "system"]
        c = anthropic.Anthropic(api_key=_secret("ANTHROPIC_API_KEY") or _secret("ANTHROPIC_API"), timeout=timeout)
        kw: dict[str, Any] = dict(model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"), max_tokens=max_tokens,
                                  temperature=temperature, messages=rest)
        if sys_txt:
            kw["system"] = sys_txt
        r = c.messages.create(**kw)
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    base, key, mdl = {
        "groq": (GROQ_BASE_URL, groq_key(), groq_model(purpose)),
        "xai": ("https://api.x.ai/v1", _secret("XAI_API_KEY"), "grok-3-mini"),
        "openai": (None, _secret("OPENAI_API_KEY"), "gpt-4o-mini"),
    }[p]
    from openai import OpenAI
    c = OpenAI(api_key=key, base_url=base, timeout=timeout, max_retries=1)
    kw = dict(model=mdl, messages=msgs, max_tokens=max_tokens, temperature=temperature)
    if p == "groq":
        kw.update(groq_extra(mdl))
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    r = c.chat.completions.create(**kw)
    return r.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------

def embedding_model() -> str:
    return {EMBED_OLLAMA: ollama_embed_model, EMBED_LOCAL: local_embed_model,
            EMBED_OPENAI: lambda: "text-embedding-3-small"}[embed_provider()]()


def embed(texts: list[str], client=None) -> list[list[float]]:
    client = client or openai_client()
    r = client.embeddings.create(model=embedding_model(), input=texts)
    return [d.embedding for d in r.data]


def embeddings_index_path(default_path):
    """Each embeddings provider has its own index (vector sizes differ):
    OpenAI keeps ``default_path``; local fastembed / Ollama use a sibling
    file (embeddings.npz -> embeddings.local.npz / embeddings.ollama.npz)."""
    from pathlib import Path
    ep = embed_provider()
    if ep == EMBED_OPENAI:
        return default_path
    p = Path(default_path)
    out = p.with_name(p.stem + "." + ep + p.suffix)
    return out if isinstance(default_path, Path) else str(out)


def describe() -> dict:
    """What the switch currently resolves to (for status pages / logs)."""
    if is_ollama():
        return {"provider": "ollama", "host": ollama_host(), "model": ollama_model(),
                "embed_provider": EMBED_OLLAMA, "embed_model": ollama_embed_model()}
    chat_p = api_chat_provider()
    return {"provider": "api", "configured": configured_api_providers(), "chat_provider": chat_p,
            "chat_model": groq_model() if chat_p == "groq" else openai_fallback_model(),
            "embed_provider": embed_provider(), "embed_model": embedding_model()}


def ollama_json(prompt: str, schema_hint: str = "") -> Any:
    """Small convenience: ask for JSON and parse it (tolerates code fences)."""
    raw = chat([{"role": "user", "content": prompt + ("\n\n" + schema_hint if schema_hint else "")}],
               json_mode=True)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)
