"""Operations Agent — day-to-day hospital operations monitoring.

Ensures facilities are running smoothly by tracking patient flow, bed
occupancy, staff scheduling, departmental throughput, and adherence
to standard operating procedures.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from agents.state import AgentState
from agents.tools.snowflake_tools import list_available_tables, query_snowflake

logger = logging.getLogger("agents")

_SYSTEM_PROMPT = """You are the Operations Agent for Afya DataHub, ensuring smooth day-to-day hospital operations across facilities in Kenya.

Responsibilities:
- Monitor patient flow, bed occupancy, staff scheduling, and departmental throughput.
- Identify operational bottlenecks and resource constraints before they escalate.
- Track adherence to standard operating procedures and clinical pathways.
- Flag issues that require immediate management attention.
- Provide actionable recommendations to improve operational efficiency.

Key metrics to track:
- Bed utilisation rate (target: 75–85%)
- Patient wait times by department
- Discharge rates and average length of stay
- Equipment availability and downtime
- Staff coverage vs. patient load ratios
- Theatre utilisation

Always compare current performance against facility targets and historical baselines.

Structure your output: Current Status → Issues Identified → Root Cause Analysis → Recommended Actions."""

_TOOLS = [query_snowflake, list_available_tables]


def operations_agent_node(state: AgentState) -> dict:
    """Assess and report on hospital operational status."""
    from agents.llm import get_llm
    from langgraph.prebuilt import create_react_agent

    llm = get_llm(temperature=0.1)
    agent = create_react_agent(llm, _TOOLS, state_modifier=_SYSTEM_PROMPT)

    prompt = (
        f"Assess operational status: {state['task']}\n\n"
        f"User role: {state.get('user_role', 'Unknown')}\n"
        f"Facility: {state.get('facility') or 'All Facilities'}"
    )

    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    output = result["messages"][-1].content
    logger.info("Operations Agent completed: %d chars", len(output))

    return {
        "agent_outputs": {"operations_agent": output},
        "messages": result["messages"],
    }
