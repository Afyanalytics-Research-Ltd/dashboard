"""
The LangGraph state machine.

    START → prime → analyst ⇄ tools → END

`prime` resets the per-turn counters. `analyst` is the LLM turn - it prepends a
fixed preamble (system prompt + the deterministic workbook profile) to the
conversation, so the model never starts blind and the schema can never be
trimmed away. `tools` executes whatever it asked for; the conditional edge
loops until the model answers without requesting a tool.

The graph is deliberately written out with `StateGraph` rather than assembled
by `langchain.agents.create_agent`. It is a handful more lines, and in return
you can see and change every transition - which is the whole reason to reach
for LangGraph on a task like this.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Annotated, Any, Sequence

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from .prompts import CONTEXT_TEMPLATE, SYSTEM_PROMPT
from .session import AnalysisSession
from .tools import build_tools

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("ANALYST_MODEL", "gpt-4.1")
#: Hard stop on the tool loop, so a confused model cannot bill you forever.
MAX_TOOL_ROUNDS = int(os.getenv("ANALYST_MAX_TOOL_ROUNDS", "12"))


class AnalystState(TypedDict, total=False):
    """Graph state. `messages` accumulates; `tool_rounds` is the loop guard."""

    messages: Annotated[list[AnyMessage], add_messages]
    tool_rounds: int
    wrapped_up: bool


@dataclass(slots=True)
class AnalystReply:
    """What one `ask()` produced."""

    text: str
    tool_calls: list[dict[str, Any]]
    artifacts: list[dict[str, str]]
    messages: list[BaseMessage]


def build_graph(
    session: AnalysisSession,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    api_key: str | None = None,
):
    """Compile a graph bound to `session`. Cheap enough to call per request."""
    tools = build_tools(session)
    llm = ChatOpenAI(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        timeout=120,
        max_retries=2,
    ).bind_tools(tools)

    #: Rebuilt once per graph, prepended to every model call. Kept OUT of
    #: `state["messages"]` so the stored transcript stays purely conversational
    #: and the system block can never drift to the wrong position, or be lost
    #: when history is trimmed.
    preamble = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=CONTEXT_TEMPLATE.format(overview=session.overview())),
    ]

    def prime(state: AnalystState) -> dict[str, Any]:
        """Reset the per-turn counters."""
        return {"tool_rounds": 0, "wrapped_up": False}

    def analyst(state: AnalystState) -> dict[str, Any]:
        response = llm.invoke([*preamble, *state["messages"]])
        return {"messages": [response]}

    tool_node = ToolNode(tools, handle_tool_errors=True)

    def route(state: AnalystState) -> str:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return END
        if state.get("wrapped_up"):
            # Already asked for a final answer once; do not loop again.
            log.warning("Model requested tools after wrap-up; ending the turn.")
            return END
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            log.warning("Tool round limit (%s) reached; forcing an answer.", MAX_TOOL_ROUNDS)
            return "wrap_up"
        return "tools"

    def count_round(state: AnalystState) -> dict[str, Any]:
        return {"tool_rounds": state.get("tool_rounds", 0) + 1}

    def wrap_up(state: AnalystState) -> dict[str, Any]:
        """Budget exhausted: satisfy the pending tool calls, then ask for a
        final answer from what has already been computed."""
        last = state["messages"][-1]
        stubs: list[AnyMessage] = [
            ToolMessage(
                content="Tool budget for this turn is exhausted.",
                tool_call_id=call["id"],
            )
            for call in getattr(last, "tool_calls", [])
        ]
        stubs.append(
            HumanMessage(
                content=(
                    "You have used the tool budget for this turn. Answer now "
                    "using only what you have already computed, and say plainly "
                    "what is still unresolved."
                )
            )
        )
        return {"messages": stubs, "wrapped_up": True}

    builder = StateGraph(AnalystState)
    builder.add_node("prime", prime)
    builder.add_node("analyst", analyst)
    builder.add_node("tools", tool_node)
    builder.add_node("count_round", count_round)
    builder.add_node("wrap_up", wrap_up)

    builder.add_edge(START, "prime")
    builder.add_edge("prime", "analyst")
    builder.add_conditional_edges(
        "analyst", route, {"tools": "tools", "wrap_up": "wrap_up", END: END}
    )
    builder.add_edge("tools", "count_round")
    builder.add_edge("count_round", "analyst")
    builder.add_edge("wrap_up", "analyst")

    return builder.compile()


# --------------------------------------------------------------------------- #
# Convenience wrapper
# --------------------------------------------------------------------------- #

def ask(
    session: AnalysisSession,
    question: str,
    *,
    history: Sequence[BaseMessage] = (),
    model: str | None = None,
    api_key: str | None = None,
) -> AnalystReply:
    """Run one turn.

    `history` is the prior transcript - conversation messages only, no system
    prompt (the graph adds its own preamble on every model call). The Django
    layer keeps that history in the database and replays it, so a restarted
    worker loses nothing and there is exactly one source of truth.
    """
    graph = build_graph(session, model=model, api_key=api_key)
    before = len(session.artifacts)

    inbound: list[AnyMessage] = [*history, HumanMessage(content=question)]
    final = graph.invoke(
        {"messages": inbound},
        config={"recursion_limit": MAX_TOOL_ROUNDS * 3 + 10},
    )

    messages: list[BaseMessage] = final["messages"]
    # Only this turn's work - `messages` also carries the replayed history.
    produced = messages[len(inbound) :]
    answer = _final_answer(produced or messages)

    tool_calls = [
        {"name": call["name"], "args": call["args"]}
        for m in produced
        if isinstance(m, AIMessage)
        for call in (m.tool_calls or [])
    ]

    return AnalystReply(
        text=answer,
        tool_calls=tool_calls,
        artifacts=[a.to_dict() for a in session.artifacts[before:]],
        messages=messages,
    )


def _message_text(message: BaseMessage) -> str:
    """Plain text of a message across content formats (string or block list)."""
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    if callable(text):  # older langchain-core exposed .text() as a method
        return str(text())
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _final_answer(messages: Sequence[BaseMessage]) -> str:
    """The last assistant message that is prose rather than a tool request.

    Falls back to any trailing assistant text, then to a neutral notice, so a
    view never has to render an empty bubble.
    """
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            text = _message_text(message).strip()
            if text:
                return text
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = _message_text(message).strip()
            if text:
                return text
    return "I could not produce an answer for that. Try narrowing the question."
