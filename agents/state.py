from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    question: str           # original natural-language question
    user_id: str            # identifier for the requesting user
    user_phone: Optional[str]  # WhatsApp-capable phone e.g. "254712345678"
    callback_url: str       # POST target for async Phase-2 result delivery
    thread_id: str          # LangGraph thread ID (mirrors configurable key)

    # ── Classification ─────────────────────────────────────────────────────
    matched_metric: Optional[dict]      # {id, name, cube_query, filters}
    classification_confidence: float    # 0.0–1.0; below threshold → fallback

    # ── Execution ──────────────────────────────────────────────────────────
    cube_query: Optional[dict]   # final Cube query dict sent to /load
    raw_result: Optional[dict]   # raw Cube API response

    # ── Output ─────────────────────────────────────────────────────────────
    formatted_result: Optional[dict]  # shaped response returned to caller

    # ── Control flow ───────────────────────────────────────────────────────
    is_resumed: bool             # True after human completes the loop
    fallback_reason: Optional[str]
    resume_data: Optional[dict]  # payload passed by human via Command(resume=…)