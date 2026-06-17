"""Communications Agent — multi-channel notification dissemination.

Distributes information and alerts across email and WhatsApp channels,
targeting the right personnel with the right level of detail.
"""
from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage

from agents.state import AgentState
from agents.tools.email_tools import send_email
from agents.tools.whatsapp_tools import send_whatsapp_message

logger = logging.getLogger("agents")

_SYSTEM_PROMPT = """You are the Communications Agent for Afya DataHub, responsible for disseminating critical information across hospital teams in Kenya.

Responsibilities:
- Send timely notifications via email and WhatsApp to the appropriate personnel.
- Select the right channel based on urgency:
  * Email: reports, approvals, summaries, non-urgent updates
  * WhatsApp: urgent alerts, stockout warnings, critical operational issues
- Tailor message content to the recipient's role:
  * Facility staff → plain language, specific action required, deadline if applicable
  * Managers → summary with key numbers, impact, and recommended decision
  * Executives → headline metrics only, strategic impact, escalation status
- Do not send duplicate notifications.
- Confirm what was sent, to whom, and via which channel in your summary.

Always be concise and actionable. Every message should answer: What happened? Why does it matter? What should the recipient do?"""

_TOOLS = [send_email, send_whatsapp_message]


def communications_agent_node(state: AgentState) -> dict:
    """Disseminate information across email and WhatsApp channels."""
    from agents.llm import get_llm
    from langgraph.prebuilt import create_react_agent

    llm = get_llm(temperature=0.2)
    agent = create_react_agent(llm, _TOOLS, state_modifier=_SYSTEM_PROMPT)

    # Include outputs from preceding agents as message context
    outputs_context = ""
    if state.get("agent_outputs"):
        outputs_context = "\n\nContext from other agents:\n" + json.dumps(
            state["agent_outputs"], indent=2, default=str
        )[:2000]

    prompt = (
        f"Disseminate the following information: {state['task']}{outputs_context}\n\n"
        f"User role: {state.get('user_role', 'Unknown')}\n"
        f"Facility: {state.get('facility') or 'All Facilities'}"
    )

    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    output = result["messages"][-1].content
    logger.info("Communications Agent completed: %d chars", len(output))

    return {
        "agent_outputs": {"communications_agent": output},
        "messages": result["messages"],
    }
