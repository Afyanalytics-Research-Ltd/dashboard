"""Client for the mcp-browserbase MCP server (docker-compose.mcp-browserbase.yaml).

Distinct from services.py: that module talks to Browserbase directly via
its Python SDK for the human-facing console (create/list/view/end). This
module instead talks to Browserbase through the MCP server's tools —
start/navigate/act/extract/observe/end — the shape an LLM agent expects
when driving a browser as part of a tool-calling loop.

Each call here opens its own Streamable HTTP connection and therefore its
own MCP session, which owns exactly one Browserbase browser session for
the duration of the connection. There is no way to attach an MCP tool
call to a Browserbase session created outside of MCP (via services.py) —
the two session lifecycles are independent by design of the upstream
mcp-browserbase server.

``act``/``extract``/``observe`` are AI-driven (Stagehand) and need a real
model key configured on the mcp-browserbase container (GEMINI_API_KEY, or
--modelName/--modelApiKey in its `command`) — without one they'll raise.
``start``/``navigate``/``end`` do not need a model key.
"""

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from typing import Any, Optional

from django.conf import settings
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r'https?://\S+')


class BrowserMCPError(Exception):
    """Raised when an MCP tool call fails or returns an error result."""


def route_message(text: str) -> tuple[str, dict]:
    """Turn one line of natural-language text into an (tool_name, args) call.

    A simple, transparent set of rules — not its own LLM call — so the
    mapping from what you type to which MCP tool runs is predictable:
      "navigate:<url>" / "go to <url>" / "open <url>" / a bare URL → navigate
      "act:<instruction>"                                          → act
      "extract:<instruction>"                                      → extract
      "observe:<instruction>"                                      → observe
      anything else                                                → act
    Shared by the browser-agent chat (consumers.py) and scheduled browser
    tasks (tasks.py), so both interpret instructions identically.
    """
    stripped = text.strip()
    lower = stripped.lower()

    for prefix, tool, key in (
        ('navigate:', 'navigate', 'url'),
        ('act:', 'act', 'action'),
        ('extract:', 'extract', 'instruction'),
        ('observe:', 'observe', 'instruction'),
    ):
        if lower.startswith(prefix):
            return tool, {key: stripped[len(prefix):].strip()}

    url_match = _URL_RE.search(stripped)
    if url_match and (
        stripped == url_match.group(0)
        or lower.startswith(('go to', 'navigate to', 'open '))
    ):
        return 'navigate', {'url': url_match.group(0)}

    return 'act', {'action': stripped}


def format_tool_reply(tool_name: str, result) -> str:
    """Render a parsed tool result as human-readable text."""
    if tool_name == 'navigate' and isinstance(result, dict):
        return f"Navigated to {result.get('url', '(unknown url)')}."
    if isinstance(result, (dict, list)):
        return f"```json\n{json.dumps(result, indent=2, default=str)}\n```"
    return str(result)


@asynccontextmanager
async def _session():
    async with streamable_http_client(settings.BROWSERBASE_MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def parse_tool_result(tool_name: str, result) -> Any:
    """Unwrap an MCP CallToolResult into the plain dict/str its JSON text encodes."""
    if result.is_error:
        raise BrowserMCPError(f"{tool_name} failed: {result.content}")
    text = "".join(
        block.text for block in result.content if getattr(block, "type", None) == "text"
    )
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    if isinstance(payload, dict) and payload.get("success") is False:
        raise BrowserMCPError(f"{tool_name} failed: {payload}")
    return payload.get("data", payload) if isinstance(payload, dict) else payload


async def list_tools() -> list[str]:
    """Return the names of every tool the MCP server exposes."""
    async with _session() as session:
        tools = await session.list_tools()
        return [t.name for t in tools.tools]


async def run_task(steps: list[dict]) -> list[Any]:
    """Run a sequence of MCP tool calls on ONE Browserbase session.

    Opens a session (calling ``start`` implicitly isn't required — the
    server creates one lazily on first tool call), runs each step, then
    always calls ``end`` — even if a step raises — so a failed task never
    leaves a session running.

    ``steps`` is a list of ``{"tool": "navigate", "args": {"url": "..."}}``
    dicts. Returns the parsed result of each step, in order.

    Example:
        results = await run_task([
            {"tool": "navigate", "args": {"url": "https://example.com"}},
            {"tool": "extract", "args": {"instruction": "get the page title"}},
        ])
    """
    results = []
    async with _session() as session:
        try:
            for step in steps:
                tool_name = step["tool"]
                args = step.get("args", {})
                result = await session.call_tool(tool_name, args)
                results.append(parse_tool_result(tool_name, result))
        finally:
            try:
                await session.call_tool("end", {})
            except Exception as exc:
                logger.warning("mcp-browserbase: failed to end session cleanly: %s", exc)
    return results


async def navigate(url: str) -> dict:
    """Convenience wrapper: start a session, go to ``url``, end the session."""
    results = await run_task([{"tool": "navigate", "args": {"url": url}}])
    return results[0]


class OpenSession:
    """A Browserbase session held open across multiple tool calls over time.

    Unlike ``run_task``/``navigate`` (open a connection, run steps, close it
    in one call), this is for callers — the browser-agent chat consumer —
    that need ONE session to persist across many separate calls, since
    mcp-browserbase ties the underlying Browserbase browser session to the
    lifetime of this MCP connection: closing it ends the browser session.

    anyio (which streamable_http_client/ClientSession are built on) requires
    its context managers to be entered AND exited within the same task, so
    this can't just hold onto them across separate calls from the consumer's
    connect()/receive()/disconnect() — instead, ONE background task owns a
    single unbroken ``async with`` block for the object's whole lifetime,
    and every call is dispatched to it over a queue and awaited via a Future.
    """

    def __init__(self):
        self._queue: "asyncio.Queue" = asyncio.Queue()
        self._ready = asyncio.Event()
        self._error: Optional[BaseException] = None
        self.browserbase_session_id: Optional[str] = None
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            async with streamable_http_client(settings.BROWSERBASE_MCP_URL) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._ready.set()
                    while True:
                        job = await self._queue.get()
                        if job is None:
                            break
                        tool_name, args, future = job
                        if future.cancelled():
                            continue
                        try:
                            result = await session.call_tool(tool_name, args)
                            future.set_result(result)
                        except Exception as exc:
                            future.set_exception(exc)
        except Exception as exc:
            self._error = exc
            self._ready.set()

    async def _wait_ready(self) -> None:
        await self._ready.wait()
        if self._error is not None:
            raise BrowserMCPError(f"Failed to open MCP session: {self._error}") from self._error

    async def call(self, tool_name: str, args: Optional[dict] = None) -> Any:
        future: "asyncio.Future" = asyncio.get_running_loop().create_future()
        await self._queue.put((tool_name, args or {}, future))
        result = await future
        return parse_tool_result(tool_name, result)

    async def close(self) -> None:
        try:
            await self.call("end")
        except Exception as exc:
            logger.warning("mcp-browserbase: failed to end session cleanly: %s", exc)
        await self._queue.put(None)
        try:
            await asyncio.wait_for(self._task, timeout=10)
        except asyncio.TimeoutError:
            logger.warning("mcp-browserbase: worker task didn't stop in time, cancelling")
            self._task.cancel()


async def open_session() -> OpenSession:
    """Open a long-lived MCP connection. Caller MUST ``await .close()`` it."""
    session = OpenSession()
    await session._wait_ready()
    return session
