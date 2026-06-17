"""Metrics Agent — continuous KPI monitoring and reporting.

Tracks performance indicators across facilities and reports findings
to appropriate users based on their roles in the role hierarchy.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from agents.state import AgentState
from agents.tools.snowflake_tools import list_available_tables, query_snowflake

logger = logging.getLogger("agents")

_SYSTEM_PROMPT = """You are the Metrics Agent for Afya DataHub, monitoring key performance indicators across hospitals in Kenya.

Responsibilities:
- Track KPIs including revenue, patient volumes, pharmacy inventory levels, staff utilisation, and clinical outcomes.
- Identify trends, anomalies, and deviations from targets.
- Produce concise, role-appropriate reports:
  * Executives → headline metrics and strategic summary
  * Facility Managers → operational detail and department breakdown
  * Analysts → raw numbers with statistical context
- Flag urgent issues clearly: stockouts, revenue drops >10%, unusual admission spikes.

Always structure your report with these sections:
1. Summary
2. Key Metrics (with values and period-over-period change)
3. Alerts (issues requiring immediate attention)
4. Recommendations"""

_TOOLS = [query_snowflake, list_available_tables]


def metrics_agent_node(state: AgentState) -> dict:
    """Monitor KPIs and generate role-appropriate performance reports."""
    from agents.llm import get_llm
    from langgraph.prebuilt import create_react_agent

    llm = get_llm(temperature=0.1)
    agent = create_react_agent(llm, _TOOLS, state_modifier=_SYSTEM_PROMPT)

    prompt = (
        f"Generate a KPI report for: {state['task']}\n\n"
        f"User role: {state.get('user_role', 'Unknown')}\n"
        f"Facility: {state.get('facility') or 'All Facilities'}"
    )

    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    output = result["messages"][-1].content
    logger.info("Metrics Agent completed: %d chars", len(output))

    return {
        "agent_outputs": {"metrics_agent": output},
        "messages": result["messages"],
    }
