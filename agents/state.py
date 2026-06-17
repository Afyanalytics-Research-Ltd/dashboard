from __future__ import annotations

import operator
from typing import Annotated, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage


def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


class AgentState(TypedDict):
    """Shared state that flows through the entire agent graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str
    user_role: str
    facility: Optional[str]
    next_agent: str
    agent_outputs: Annotated[dict, _merge_dicts]
    final_response: str
    iteration_count: int
    evaluation: Optional[str]
