"""SQL Agent — read-only access to the Snowflake semantic layer.

Provides structured query access to the data warehouse without exposing
direct database credentials or permitting destructive operations.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from agents.state import AgentState
from agents.tools.snowflake_tools import (
    get_table_sample,
    list_available_tables,
    query_snowflake,
)

logger = logging.getLogger("agents")

_SYSTEM_PROMPT = """You are the SQL Agent for Afya DataHub, a healthcare analytics platform serving hospitals in Kenya.

Your sole responsibility is to retrieve accurate data from the Snowflake data warehouse.

Guidelines:
- Call list_available_tables first when you are unsure which tables exist.
- Use get_table_sample to inspect a table's columns before writing a complex query.
- Write clean, efficient SELECT statements targeting the HOSPITALS database.
- Limit result sets to what is relevant — avoid fetching thousands of rows unnecessarily.
- If a query fails, explain the error and try an alternative approach.
- You have read-only access — you cannot modify any data.

Return a clear, structured summary of findings with key numbers highlighted."""

_TOOLS = [query_snowflake, list_available_tables, get_table_sample]


def sql_agent_node(state: AgentState) -> dict:
    """Execute data retrieval tasks against Snowflake."""
    from agents.llm import get_llm
    from langgraph.prebuilt import create_react_agent

    llm = get_llm(temperature=0.0)
    agent = create_react_agent(llm, _TOOLS, state_modifier=_SYSTEM_PROMPT)

    result = agent.invoke({"messages": [HumanMessage(content=state["task"])]})
    output = result["messages"][-1].content
    logger.info("SQL Agent completed: %d chars", len(output))

    return {
        "agent_outputs": {"sql_agent": output},
        "messages": result["messages"],
    }
