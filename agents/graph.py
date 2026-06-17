"""Main LangGraph multi-agent graph for Afya DataHub.

Architecture:
    START → supervisor → [execution agent] → supervisor → … → END

The supervisor sits at the top of the hierarchy and routes tasks to the
appropriate execution agent. After each agent completes, control returns
to the supervisor which evaluates the output and decides next steps.

Routing:
    supervisor.next_agent == "FINISH"   → END
    supervisor.next_agent == "<name>"   → that agent node → supervisor
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from agents.nodes.communications_agent import communications_agent_node
from agents.nodes.metrics_agent import metrics_agent_node
from agents.nodes.operations_agent import operations_agent_node
from agents.nodes.procurement_agent import procurement_agent_node
from agents.nodes.sql_agent import sql_agent_node
from agents.nodes.supervisor import supervisor_node
from agents.state import AgentState

logger = logging.getLogger("agents")

_EXECUTION_AGENTS = [
    "sql_agent",
    "metrics_agent",
    "procurement_agent",
    "operations_agent",
    "communications_agent",
]


def _route_from_supervisor(state: AgentState) -> str:
    """Conditional edge: read supervisor's routing decision."""
    next_agent = state.get("next_agent", "FINISH")
    if next_agent not in _EXECUTION_AGENTS:
        return END
    return next_agent


def build_graph():
    """Build and compile the supervisor-controlled multi-agent graph."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("metrics_agent", metrics_agent_node)
    graph.add_node("procurement_agent", procurement_agent_node)
    graph.add_node("operations_agent", operations_agent_node)
    graph.add_node("communications_agent", communications_agent_node)

    # Graph always starts at the supervisor
    graph.add_edge(START, "supervisor")

    # Each execution agent returns control to the supervisor
    for agent in _EXECUTION_AGENTS:
        graph.add_edge(agent, "supervisor")

    # Supervisor routes conditionally to agents or END
    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "sql_agent": "sql_agent",
            "metrics_agent": "metrics_agent",
            "procurement_agent": "procurement_agent",
            "operations_agent": "operations_agent",
            "communications_agent": "communications_agent",
            END: END,
        },
    )

    return graph.compile()


# Module-level singleton — compiled once, reused across requests
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agents(
    task: str,
    user_role: str = "Facility Admin",
    facility: str | None = None,
) -> dict:
    """Invoke the multi-agent graph and return the final response.

    Args:
        task: The task, question, or instruction for the agents.
        user_role: Role of the requesting user (from authentication.roles).
        facility: Optional facility name for data-scoping context.

    Returns:
        Dict with keys: final_response, agent_outputs, evaluation, iterations.
    """
    graph = get_graph()

    initial_state: AgentState = {
        "messages": [],
        "task": task,
        "user_role": user_role,
        "facility": facility,
        "next_agent": "",
        "agent_outputs": {},
        "final_response": "",
        "iteration_count": 0,
        "evaluation": None,
    }

    logger.info(
        "Agent graph invoked | task=%.80s | role=%s | facility=%s",
        task,
        user_role,
        facility or "all",
    )

    try:
        result = graph.invoke(initial_state)
        return {
            "final_response": result.get("final_response", ""),
            "agent_outputs": result.get("agent_outputs", {}),
            "evaluation": result.get("evaluation", ""),
            "iterations": result.get("iteration_count", 0),
        }
    except Exception as exc:
        logger.error("Agent graph error: %s", exc, exc_info=True)
        return {
            "final_response": f"An error occurred while processing your request: {exc}",
            "agent_outputs": {},
            "evaluation": "",
            "iterations": 0,
        }
