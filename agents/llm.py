"""LangChain LLM factory — mirrors the priority order of ai_client.py.

Priority: Groq (free) → Grok (xAI) → Claude (Anthropic)
All returned models support tool calling for use with create_react_agent.
"""
from __future__ import annotations

import os
from typing import Any


def get_llm(temperature: float = 0.3) -> Any:
    """Return a LangChain chat model using environment-based provider selection."""
    if os.getenv("GROQ_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            # 70b model is required for reliable tool calling on Groq
            model="llama-3.3-70b-versatile",
            temperature=temperature,
        )

    if os.getenv("XAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
            model="grok-3-mini",
            temperature=temperature,
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-haiku-4-5",
            temperature=temperature,
        )

    raise RuntimeError(
        "No LLM API key configured. Set GROQ_API_KEY, XAI_API_KEY, or ANTHROPIC_API_KEY."
    )
