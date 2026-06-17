"""Supervisor Agent — quality control and routing layer.

Sits above all execution agents in the hierarchy. Evaluates incoming tasks,
routes them to the most appropriate agent, and assesses output quality before
deciding to route further or declare the task complete.
"""
from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState

logger = logging.getLogger("agents")

_AGENTS = {
    "sql_agent",
    "metrics_agent",
    "procurement_agent",
    "operations_agent",
    "communications_agent",
    "FINISH",
}

_SYSTEM_PROMPT = """You are the Supervisor Agent for Afya DataHub, a healthcare analytics platform serving hospitals in Kenya.

Your role:
1. Evaluate incoming tasks and route them to the most appropriate execution agent.
2. Assess completed agent outputs for accuracy, completeness, and quality.
3. Decide whether to call another agent for additional work, or declare FINISH.

Execution agents and their responsibilities:
- sql_agent: Queries the Snowflake data warehouse. Use for any data retrieval or reporting need.
- metrics_agent: Monitors KPIs and performance indicators. Use for performance reviews and trend reports.
- procurement_agent: Handles approval requests and supplier communications. Use for purchase orders.
- operations_agent: Monitors day-to-day hospital operations. Use for operational status and workflow issues.
- communications_agent: Sends notifications via email and WhatsApp. Use when information must be disseminated.
- FINISH: Task is complete — the accumulated agent outputs are sufficient.

User context — role: {user_role} | facility: {facility}

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{{
  "next_agent": "<agent name or FINISH>",
  "reasoning": "<one sentence explanation>",
  "evaluation": "<quality assessment of completed work, or N/A if first routing>"
}}"""


def supervisor_node(state: AgentState) -> dict:
    """Route tasks to agents and evaluate their outputs."""
    iteration = state.get("iteration_count", 0)

    # Hard stop to prevent runaway loops
    if iteration >= 5:
        logger.warning("Supervisor: max iterations reached, forcing FINISH")
        return {
            "next_agent": "FINISH",
            "evaluation": "Max iterations reached.",
            "final_response": _build_final_response(state),
            "iteration_count": iteration + 1,
        }

    from agents.llm import get_llm
    llm = get_llm(temperature=0.1)

    system = _SYSTEM_PROMPT.format(
        user_role=state.get("user_role", "Unknown"),
        facility=state.get("facility") or "All Facilities",
    )

    outputs_summary = ""
    if state.get("agent_outputs"):
        outputs_summary = "\n\nCompleted agent outputs:\n" + json.dumps(
            state["agent_outputs"], indent=2, default=str
        )[:3000]

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Task: {state['task']}{outputs_summary}"),
    ]

    decision = {"next_agent": "FINISH", "reasoning": "default", "evaluation": "N/A"}
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        # Strip markdown code fences if present
        if "```" in content:
            parts = content.split("```")
            content = parts[1] if len(parts) > 1 else parts[0]
            if content.startswith("json"):
                content = content[4:]
        decision = json.loads(content.strip())
    except Exception as exc:
        logger.error("Supervisor LLM error: %s", exc)
        # Default: route to sql_agent on first call, FINISH thereafter
        decision["next_agent"] = "sql_agent" if not state.get("agent_outputs") else "FINISH"

    next_agent = decision.get("next_agent", "FINISH")
    if next_agent not in _AGENTS:
        logger.warning("Supervisor returned unknown agent '%s', defaulting to FINISH", next_agent)
        next_agent = "FINISH"

    logger.info("Supervisor → %s (iter=%d)", next_agent, iteration)

    updates: dict = {
        "next_agent": next_agent,
        "evaluation": decision.get("evaluation", ""),
        "iteration_count": iteration + 1,
    }
    if next_agent == "FINISH":
        updates["final_response"] = _build_final_response(state)

    return updates


def _build_final_response(state: AgentState) -> str:
    """Synthesise accumulated agent outputs into a single response."""
    outputs = state.get("agent_outputs", {})
    if not outputs:
        return "No results were produced for this task."

    parts = [f"**Task:** {state.get('task', '')}\n"]
    for agent_name, output in outputs.items():
        label = agent_name.replace("_", " ").title()
        parts.append(f"**{label}:**\n{output}\n")

    evaluation = state.get("evaluation", "")
    if evaluation and evaluation != "N/A":
        parts.append(f"**Quality Assessment:** {evaluation}")

    return "\n".join(parts)
