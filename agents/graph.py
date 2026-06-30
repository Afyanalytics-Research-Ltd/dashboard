"""
LangGraph graph assembly.

Graph topology:

  START
    │
    ▼
  classify_intent
    │
    ├─[match]──► execute_query ──► format_result ──► END
    │
    └─[no match]──► fallback_notifier ──[interrupt]──► re_classify ──► execute_query ──► format_result
                                                                                              │
                                                                                              ▼
                                                                                        auto_notify_user ──► END

Note: execute_query and format_result are shared between both paths.
      A conditional edge after format_result routes to END (happy) or
      auto_notify_user (resumed), keyed on state["is_resumed"].
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command  # re-exported for callers

from .nodes import (
    auto_notify_user,
    classify_intent,
    execute_query,
    fallback_notifier,
    format_result,
    re_classify,
)
from .state import AgentState


# ── Conditional edges ─────────────────────────────────────────────────────────

def _route_after_classify(state: AgentState) -> str:
    """Route to execute_query if a metric matched, otherwise to fallback."""
    return "execute_query" if state.get("matched_metric") else "fallback_notifier"


def _route_after_format(state: AgentState) -> str:
    """
    After formatting:
      - Happy path  → END (response already in state, returned via HTTP).
      - Resume path → auto_notify_user (push result to client webhook).
    """
    return "auto_notify_user" if state.get("is_resumed") else END


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("execute_query", execute_query)
    builder.add_node("format_result", format_result)
    builder.add_node("fallback_notifier", fallback_notifier)
    builder.add_node("re_classify", re_classify)
    builder.add_node("auto_notify_user", auto_notify_user)

    # Edges
    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges("classify_intent", _route_after_classify)

    # Happy path
    builder.add_edge("execute_query", "format_result")
    builder.add_conditional_edges("format_result", _route_after_format)

    # Fallback / resume path
    builder.add_edge("fallback_notifier", "re_classify")
    builder.add_edge("re_classify", "execute_query")
    builder.add_edge("auto_notify_user", END)

    # Compile with in-memory checkpointer (swap for PostgresSaver in production)
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Singleton — import this in views
graph = build_graph()