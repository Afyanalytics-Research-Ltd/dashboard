"""api-mode chat routing: OpenAI-SDK call sites go to Groq (open-source
models) when a Groq key exists, with OpenAI as fallback; embeddings stay
on OpenAI."""
from unittest import mock

import httpx
import openai
import pytest

from afya_backends import llm, toggles


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for k in ("AFYA_LLM_PROVIDER", "LLM_PROVIDER_DEFAULT", "LLM_API_CHAT_PROVIDER", "GROQ_API", "GROQ_MODEL",
              "GROQ_MODEL_EXTRACTION", "OPENAI_FALLBACK_MODEL", "ANTHROPIC_API_KEY", "ANTHROPIC_API",
              "XAI_API_KEY", "LLM_EMBED_PROVIDER", "LOCAL_EMBED_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "_secret", lambda name: __import__("os").getenv(name, "").strip())
    saved = list(toggles._sources)
    toggles._sources.clear()
    toggles.clear_cache()
    yield
    toggles._sources[:] = saved
    toggles.clear_cache()


def _rate_limit():
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return openai.RateLimitError("rate limited", response=httpx.Response(429, request=req), body=None)


def test_model_maps_openai_names_to_groq(monkeypatch):
    assert llm.api_chat_provider() == "groq"
    assert llm.model("gpt-4o-mini") == "openai/gpt-oss-120b"
    assert llm.model("o3-mini") == "openai/gpt-oss-120b"
    assert llm.model(None) == "openai/gpt-oss-120b"
    assert llm.model("qwen/qwen3-32b") == "qwen/qwen3-32b"  # explicit non-OpenAI name kept
    monkeypatch.setenv("GROQ_MODEL_EXTRACTION", "openai/gpt-oss-120b")
    assert llm.model("gpt-4o", purpose="extraction") == "openai/gpt-oss-120b"


def test_no_groq_key_or_forced_openai_keeps_openai(monkeypatch):
    monkeypatch.setenv("LLM_API_CHAT_PROVIDER", "openai")
    assert llm.model("gpt-4o-mini") == "gpt-4o-mini"
    c = llm.openai_client()
    assert isinstance(c.embeddings, llm._LocalEmbeddings)  # embeddings still local
    assert c.chat is c._openai_or_none().chat               # chat straight to OpenAI
    monkeypatch.setenv("LLM_EMBED_PROVIDER", "openai")
    assert isinstance(llm.openai_client(), openai.OpenAI)   # nothing routed: plain client
    monkeypatch.delenv("LLM_API_CHAT_PROVIDER")
    monkeypatch.delenv("GROQ_API_KEY")
    assert llm.api_chat_provider() == "openai"
    assert llm.model("gpt-4o-mini") == "gpt-4o-mini"


def test_ollama_mode_unchanged(monkeypatch):
    monkeypatch.setenv("AFYA_LLM_PROVIDER", "ollama")
    assert llm.model("gpt-4o-mini") == llm.ollama_model()
    c = llm.openai_client()
    assert isinstance(c, openai.OpenAI) and "11434" in str(c.base_url)


def test_explicit_base_url_bypasses_routing():
    c = llm.openai_client(base_url="https://example.test/v1")
    assert isinstance(c, openai.OpenAI) and "example.test" in str(c.base_url)


def test_client_chat_goes_to_groq():
    c = llm.openai_client(api_key="sk-explicit")
    assert "groq.com" in str(c._groq.base_url)
    with mock.patch.object(c._groq.chat.completions, "create", return_value="groq-reply") as g:
        assert c.chat.completions.create(model="openai/gpt-oss-120b", messages=[]) == "groq-reply"
        g.assert_called_once()
        assert g.call_args.kwargs["reasoning_effort"] == "low"  # gpt-oss: low effort by default


def test_openai_embeddings_when_forced(monkeypatch):
    monkeypatch.setenv("LLM_EMBED_PROVIDER", "openai")
    c = llm.openai_client(api_key="sk-explicit")
    emb = c.embeddings
    assert "openai.com" in str(c._openai.base_url) and c._openai.api_key == "sk-explicit"
    assert hasattr(emb, "create")
    assert llm.embedding_model() == "text-embedding-3-small"
    assert str(llm.embeddings_index_path("catalog/embeddings.npz")) == "catalog/embeddings.npz"


class _FakeModel:
    def embed(self, texts):
        return [[float(len(t)), 1.0] for t in texts]


def test_local_embeddings_default(monkeypatch):
    assert llm.embed_provider() == "local"
    assert llm.embedding_model() == "BAAI/bge-small-en-v1.5"
    assert str(llm.embeddings_index_path("catalog/embeddings.npz")) == "catalog/embeddings.local.npz"
    monkeypatch.setattr(llm, "_local_text_embedding", lambda name: _FakeModel())
    c = llm.openai_client()
    r = c.embeddings.create(model=llm.embedding_model(), input=["ab", "abcd"])
    assert [d.embedding for d in r.data] == [[2.0, 1.0], [4.0, 1.0]]
    assert llm.embed(["xyz"]) == [[3.0, 1.0]]
    import asyncio
    ac = llm.async_openai_client()
    r = asyncio.run(ac.embeddings.create(model=llm.embedding_model(), input="hello"))
    assert r.data[0].embedding == [5.0, 1.0]


def test_ollama_embeddings_unchanged(monkeypatch):
    monkeypatch.setenv("AFYA_LLM_PROVIDER", "ollama")
    assert llm.embed_provider() == "ollama"
    assert str(llm.embeddings_index_path("catalog/embeddings.npz")) == "catalog/embeddings.ollama.npz"


def test_rate_limit_falls_back_to_openai():
    c = llm.openai_client()
    oc = c._openai_or_none()
    with mock.patch.object(c._groq.chat.completions, "create", side_effect=_rate_limit()), \
         mock.patch.object(oc.chat.completions, "create", return_value="openai-reply") as o:
        assert c.chat.completions.create(model="openai/gpt-oss-120b", messages=[], tools=[], reasoning_effort="low") == "openai-reply"
        assert o.call_args.kwargs["model"] == "gpt-4o-mini"
        assert o.call_args.kwargs["tools"] == []
        assert "reasoning_effort" not in o.call_args.kwargs  # gpt-4o-mini rejects it


def test_bad_request_is_not_masked_by_fallback():
    c = llm.openai_client()
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    bad = openai.BadRequestError("bad", response=httpx.Response(400, request=req), body=None)
    with mock.patch.object(c._groq.chat.completions, "create", side_effect=bad):
        with pytest.raises(openai.BadRequestError):
            c.chat.completions.create(model="x", messages=[])


def test_no_openai_key_reraises_groq_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    c = llm.openai_client()
    with mock.patch.object(c._groq.chat.completions, "create", side_effect=_rate_limit()):
        with pytest.raises(openai.RateLimitError):
            c.chat.completions.create(model="x", messages=[])


def test_async_client_falls_back():
    import asyncio
    c = llm.async_openai_client()
    oc = c._openai_or_none()
    with mock.patch.object(c._groq.chat.completions, "create", new=mock.AsyncMock(side_effect=_rate_limit())), \
         mock.patch.object(oc.chat.completions, "create", new=mock.AsyncMock(return_value="ok")):
        assert asyncio.run(c.chat.completions.create(model="m", messages=[])) == "ok"


def test_chat_helper_falls_through_providers():
    calls = []

    def one(p, *a, **kw):
        calls.append(p)
        if p == "groq":
            raise _rate_limit()
        return f"from {p}"
    with mock.patch.object(llm, "_chat_one", side_effect=one):
        assert llm.chat([{"role": "user", "content": "hi"}]) == "from openai"
    assert calls == ["groq", "openai"]
