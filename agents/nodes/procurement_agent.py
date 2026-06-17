"""Procurement Agent — approval requests and supplier communication.

Handles purchase order workflows, facilitates supplier contact on behalf
of appropriate teams, and escalates critical stockout situations.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from agents.state import AgentState
from agents.tools.email_tools import send_email
from agents.tools.snowflake_tools import list_available_tables, query_snowflake

logger = logging.getLogger("agents")

_SYSTEM_PROMPT = """You are the Procurement Agent for Afya DataHub, facilitating supply chain operations for hospitals in Kenya.

Responsibilities:
- Review and process procurement approval requests based on current inventory levels and budget data.
- Communicate with suppliers via email to request quotes, confirm orders, or escalate delivery issues.
- Escalate critical stockout situations immediately when stock falls below safety levels.
- Provide clear, data-backed justification for all procurement recommendations.

When sending supplier emails:
- Be professional, concise, and specific.
- Include: item name, quantity required, facility name, urgency level, and contact details.
- Copy the facility manager when corresponding with external suppliers.

Only send emails when explicitly authorised by the task or when a critical stockout requires immediate action.

Always query Snowflake first to verify current stock levels before making procurement decisions."""

_TOOLS = [query_snowflake, list_available_tables, send_email]


def procurement_agent_node(state: AgentState) -> dict:
    """Process procurement requests and manage supplier communications."""
    from agents.llm import get_llm
    from langgraph.prebuilt import create_react_agent

    llm = get_llm(temperature=0.1)
    agent = create_react_agent(llm, _TOOLS, state_modifier=_SYSTEM_PROMPT)

    prompt = (
        f"Process procurement request: {state['task']}\n\n"
        f"User role: {state.get('user_role', 'Unknown')}\n"
        f"Facility: {state.get('facility') or 'All Facilities'}"
    )

    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    output = result["messages"][-1].content
    logger.info("Procurement Agent completed: %d chars", len(output))

    return {
        "agent_outputs": {"procurement_agent": output},
        "messages": result["messages"],
    }
