"""
LLM provider abstraction.
Priority order: Groq (free tier) → Grok (xAI) → Claude (Anthropic) → None (rule-based fallback)
"""
from __future__ import annotations
import os
from typing import Optional


def _secret(key: str) -> Optional[str]:

    try:
        import streamlit as st
        # 1. Top-level
        val = st.secrets.get(key)
        if val:
            return str(val)
        # 2. Any nested section (handles key accidentally placed inside [snowflake] etc.)
        for section in st.secrets.values():
            if hasattr(section, "get"):
                val = section.get(key)
                if val:
                    return str(val)
    except Exception:
        pass
    # 3. Environment variable
    return os.getenv(key)


def get_provider() -> str:
    """Returns 'groq', 'grok', 'claude', or 'none'."""
    if _secret("GROQ_API_KEY"):
        return "groq"
    if _secret("XAI_API_KEY"):
        return "grok"
    if _secret("ANTHROPIC_API_KEY"):
        return "claude"
    return "none"


_last_error: Optional[str] = None   # exposed for sidebar debug display


def complete(
    user_prompt: str,
    system_prompt: str = "",
    max_tokens: int = 300,
) -> Optional[str]:
    """
    Call the active LLM. Returns response text or None.
    On failure, stores the error in _last_error for debugging — never raises.
    """
    global _last_error
    _last_error = None
    provider = get_provider()
    try:
        if provider == "groq":
            return _groq(user_prompt, system_prompt, max_tokens)
        if provider == "grok":
            return _grok(user_prompt, system_prompt, max_tokens)
        if provider == "claude":
            return _claude(user_prompt, system_prompt, max_tokens)
    except Exception as e:
        _last_error = f"{type(e).__name__}: {e}"
    return None


def last_error() -> Optional[str]:
    """Return the last API error, or None if the last call succeeded."""
    return _last_error


def _groq(user_prompt: str, system_prompt: str, max_tokens: int) -> str:
    from openai import OpenAI  # pip install openai
    client = OpenAI(api_key=_secret("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    r = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=msgs,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return r.choices[0].message.content.strip()


def _grok(user_prompt: str, system_prompt: str, max_tokens: int) -> str:
    from openai import OpenAI  # pip install openai
    client = OpenAI(api_key=_secret("XAI_API_KEY"), base_url="https://api.x.ai/v1")
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    r = client.chat.completions.create(
        model="grok-3-mini",
        messages=msgs,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return r.choices[0].message.content.strip()


def _claude(user_prompt: str, system_prompt: str, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=_secret("ANTHROPIC_API_KEY"))
    kwargs: dict = {
        "model": "claude-haiku-4-5",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    msg = client.messages.create(**kwargs)
    return msg.content[0].text.strip()
