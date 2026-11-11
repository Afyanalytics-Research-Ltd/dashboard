"""LangChain LLM factory — mirrors the priority order of ai_client.py.

Follows the platform LLM switch (System Settings → LLM provider):
  * "ollama": the local Ollama model (OLLAMA_MODEL, via its OpenAI API);
  * "api":    Groq (free) → Grok (xAI) → Claude (Anthropic), by which API
              keys are configured.
All returned models support tool calling for use with create_react_agent.
"""
from __future__ import annotations

import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.3) -> Any:
    """Return a LangChain chat model for the active LLM provider."""
    from afya_backends import llm as afya_llm
    if afya_llm.is_ollama():
        return ChatOpenAI(
            api_key="ollama",
            base_url=afya_llm.ollama_host() + "/v1",
            model=afya_llm.ollama_model("agent"),
            temperature=temperature,
        )

    if os.getenv("GROQ_API_KEY"):
        # A large tool-calling model (GROQ_MODEL, default openai/gpt-oss-120b;
        # llama-3.3-70b-versatile was retired by Groq).
        model = afya_llm.groq_model("agent")
        return ChatOpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url=afya_llm.GROQ_BASE_URL,
            model=model,
            temperature=temperature,
            **afya_llm.groq_extra(model),
        )

    if os.getenv("XAI_API_KEY"):
        return ChatOpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
            model="grok-3-mini",
            temperature=temperature,
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        return ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-haiku-4-5",
            temperature=temperature,
        )

    raise RuntimeError(
        "No LLM API key configured. Set GROQ_API_KEY, XAI_API_KEY, or ANTHROPIC_API_KEY "
        "(or switch System Settings → LLM provider to Ollama)."
    )
