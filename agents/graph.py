"""
LangGraph graph assembly.

Graph topology:

  START
    │
    ▼
  plan_intent            (LLM: question -> structured intent_plan)
    │
    ▼
  retrieve_candidates     (embeddings search: metrics + measures + dimensions + glossary)
    │
    ├─[confident single hit]──────────► generate_cube_query
    │
    ├─[relevant but not a single confident hit]─► compose_derived_metric
    │                                                   │
    │                              ┌─[composed a derived metric]──► generate_cube_query
    │                              ├─[cross-cube join attempted,
    │                              │  nothing computed this turn]─► explain_result (honest "connected"/"flagged" note)
    │                              └─[nothing composable]─────────► fallback_notifier
    │
    └─[nothing relevant]──────────► fallback_notifier ──[interrupt]──► re_classify ──┐
                                                                                       │
  generate_cube_query ──► validate_query ──┬─[usable query]──► execute_query ──► explain_result
                                            └─[nothing left]────────────────────────────┘
                                                                                       │
                                                                execute_query ◄────────┘
                                                                     │
                                                                     ▼
                                                               explain_result
explain_result ──[is_resumed]──► auto_notify_user ──► END
explain_result ──[not is_resumed]──► END

Note: a conditional edge after explain_result routes to END (happy path) or
      auto_notify_user (resumed path), keyed on state["is_resumed"].

Phase 4 (derived metrics) composes same-cube ratios/differences either from
a curated glossary formula or a small LLM call over two same-cube measure
candidates — never by combining separately-queried result sets, see
agents/derived_metrics.py and agents/expr_eval.py for why.

Phase 5 (cross-cube auto-join) is invoked BY compose_derived_metric, not a
separate graph node — the join must exist before generate_cube_query can
reference the joined measure. It never answers in the same turn a join is
written (Cube's reload timing isn't guaranteed same-request); explain_result
gives an honest "connected, ask again shortly" or "flagged to the analytics
team" message instead. See agents/schema_writer.py.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command  # re-exported for callers

from .derived_metrics import compose_derived_metric
from .nodes import (
    auto_notify_user,
    fallback_notifier,
    re_classify,
)
from .nodes_intent import plan_intent, retrieve_candidates
from .nodes_query import (
    CONFIDENT_SINGLE_SCORE,
    RELEVANT_FLOOR,
    find_first_resolvable,
    generate_cube_query,
    validate_query,
    execute_query,
)
from .nodes_explain import explain_result
from .state import AgentState


# ── Conditional edges ─────────────────────────────────────────────────────────

def _relevant_candidates(state: AgentState) -> list[dict]:
    candidates = state.get("retrieval_candidates") or []
    return [c for c in candidates if c.get("score", 0) >= RELEVANT_FLOOR]


def _route_after_retrieval(state: AgentState) -> str:
    """
    Route to generate_cube_query on a single confident, directly-resolvable
    hit (scanning the whole ranked list, not just the top entry — a
    dimension or formula-only glossary hit can easily outrank a perfectly
    good directly-resolvable candidate a few places down). Otherwise, if
    anything relevant exists at all, give compose_derived_metric a chance
    before giving up.
    """
    relevant = _relevant_candidates(state)
    if not relevant:
        return "fallback_notifier"
    if find_first_resolvable(relevant, CONFIDENT_SINGLE_SCORE):
        return "generate_cube_query"
    return "compose_derived_metric"


def _route_after_compose(state: AgentState) -> str:
    """
    generate_cube_query when a derived metric was actually composed;
    explain_result directly (for an honest "connected"/"flagged" note, no
    result to summarize) when a cross-cube join was attempted but nothing
    is computed this turn; fallback_notifier when nothing was composable
    at all.
    """
    if state.get("derived_metric"):
        return "generate_cube_query"
    if state.get("pending_join_writes"):
        return "explain_result"
    return "fallback_notifier"


def _route_after_validation(state: AgentState) -> str:
    """
    Route to execute_query normally; straight to explain_result (for a
    templated recovery message) if validation stripped the query down to
    no usable measures, rather than calling Cube with an empty query.
    """
    query = state.get("cube_query")
    if not query or not query.get("measures"):
        return "explain_result"
    return "execute_query"


def _route_after_explain(state: AgentState) -> str:
    """
    Happy path  → END (response already in state, returned via HTTP).
    Resume path → auto_notify_user (push result to client webhook).
    """
    return "auto_notify_user" if state.get("is_resumed") else END


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("plan_intent", plan_intent)
    builder.add_node("retrieve_candidates", retrieve_candidates)
    builder.add_node("compose_derived_metric", compose_derived_metric)
    builder.add_node("generate_cube_query", generate_cube_query)
    builder.add_node("validate_query", validate_query)
    builder.add_node("execute_query", execute_query)
    builder.add_node("explain_result", explain_result)
    builder.add_node("fallback_notifier", fallback_notifier)
    builder.add_node("re_classify", re_classify)
    builder.add_node("auto_notify_user", auto_notify_user)

    # Edges
    builder.add_edge(START, "plan_intent")
    builder.add_edge("plan_intent", "retrieve_candidates")
    builder.add_conditional_edges("retrieve_candidates", _route_after_retrieval)
    builder.add_conditional_edges("compose_derived_metric", _route_after_compose)

    # Confident-match / derived-metric path
    builder.add_edge("generate_cube_query", "validate_query")
    builder.add_conditional_edges("validate_query", _route_after_validation)

    # Shared tail
    builder.add_edge("execute_query", "explain_result")
    builder.add_conditional_edges("explain_result", _route_after_explain)

    # Fallback / resume path
    builder.add_edge("fallback_notifier", "re_classify")
    builder.add_edge("re_classify", "execute_query")
    builder.add_edge("auto_notify_user", END)

    # Compile with in-memory checkpointer (swap for PostgresSaver in production)
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Singleton — import this in views
graph = build_graph()
