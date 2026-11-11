"""LangChain LLM factory — mirrors the priority order of ai_client.py.

Follows the platform LLM switch (System Settings → LLM provider):
  * "api":    Groq (free) → Grok (xAI) → Claude (Anthropic) → OpenAI, by
              which API keys are configured;
  * "ollama": the local Ollama model (OLLAMA_MODEL, via its OpenAI API).
All returned models support tool calling for use with create_react_agent.
The provider logic lives in afya_backends.llm so pipelines share it.
"""
from __future__ import annotations

from typing import Any

from afya_backends import llm as _llm


def get_llm(temperature: float = 0.3) -> Any:
    """Return a LangChain chat model for the active LLM provider."""
    return _llm.langchain_chat(temperature=temperature)
