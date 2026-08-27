"""
The bridge between Django and the `warehouse.agent` package.

Two jobs:

1. Keep a per-conversation `AnalysisSession` warm in the worker process, so a
   50 MB workbook is not re-parsed on every message and the notebook namespace
   survives between turns.
2. Persist each turn - transcript, display messages, artifact files - inside a
   transaction, so a crash mid-turn cannot leave a half-written conversation.

Worker note: the kernel cache is process-local. Under gunicorn with several
workers, consecutive messages may land on different processes and lose
locally-defined variables (the DataFrames and transcript are always intact).
Run `--workers 1 --threads 8`, use sticky sessions, or accept that the model
occasionally recomputes an intermediate. That is the honest trade-off of
holding a live pandas kernel in a web process.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings
from django.core.files import File
from django.db import transaction
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from langchain_core.messages.utils import trim_messages

from warehouse.agent.graph import ask as run_agent
from warehouse.agent.session import AnalysisSession

from ..models import Artifact, ChatMessage, Conversation, Workbook

log = logging.getLogger(__name__)

#: How long an idle kernel is kept before it is dropped.
SESSION_TTL_SECONDS = getattr(settings, "ANALYST_SESSION_TTL", 30 * 60)
#: Maximum kernels held per worker process.
SESSION_CACHE_SIZE = getattr(settings, "ANALYST_SESSION_CACHE_SIZE", 8)
#: Messages replayed to the model. Older turns are dropped from the tail up.
MAX_HISTORY_MESSAGES = getattr(settings, "ANALYST_MAX_HISTORY_MESSAGES", 40)


# --------------------------------------------------------------------------- #
# Kernel cache
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class _Entry:
    session: AnalysisSession
    touched_at: float


_cache: dict[str, _Entry] = {}
_lock = threading.Lock()


def _artifact_dir(conversation_id: str) -> Path:
    root = Path(settings.MEDIA_ROOT) / "warehouse" / "analyst" / "scratch" / str(conversation_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _evict_locked() -> None:
    now = time.monotonic()
    for key in [k for k, e in _cache.items() if now - e.touched_at > SESSION_TTL_SECONDS]:
        _cache.pop(key, None)
    while len(_cache) > SESSION_CACHE_SIZE:
        oldest = min(_cache, key=lambda k: _cache[k].touched_at)
        _cache.pop(oldest, None)


def get_session(conversation: Conversation) -> AnalysisSession:
    """Return the warm kernel for this conversation, loading it if needed."""
    key = str(conversation.id)
    with _lock:
        entry = _cache.get(key)
        if entry is not None:
            entry.touched_at = time.monotonic()
            return entry.session

    # Load outside the lock - parsing a large workbook can take seconds.
    session = AnalysisSession.open(
        conversation.workbook.file.path,
        _artifact_dir(conversation.id),
        exec_timeout=getattr(settings, "ANALYST_EXEC_TIMEOUT", 30.0),
    )
    with _lock:
        _cache[key] = _Entry(session=session, touched_at=time.monotonic())
        _evict_locked()
    return session


def drop_session(conversation_id: str) -> None:
    with _lock:
        _cache.pop(str(conversation_id), None)


# --------------------------------------------------------------------------- #
# Workbook intake
# --------------------------------------------------------------------------- #

def profile_workbook(workbook: Workbook) -> Workbook:
    """Parse the upload once and cache its profile on the row.

    Failures are stored rather than raised, so a corrupt file shows the user a
    message instead of a 500.
    """
    try:
        session = AnalysisSession.open(
            workbook.file.path, _artifact_dir(f"probe-{workbook.id}")
        )
        workbook.overview = session.overview()
        workbook.load_error = ""
    except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
        log.exception("Failed to read workbook %s", workbook.id)
        workbook.overview = ""
        workbook.load_error = f"{type(exc).__name__}: {exc}"
    workbook.save(update_fields=["overview", "load_error"])
    return workbook


# --------------------------------------------------------------------------- #
# One turn
# --------------------------------------------------------------------------- #

def _history(conversation: Conversation) -> list[BaseMessage]:
    """Replay the stored transcript, trimmed to a sane length.

    `trim_messages(start_on="human")` keeps tool-call/tool-result pairs intact -
    orphaning a ToolMessage makes the OpenAI API reject the whole request.
    """
    if not conversation.transcript:
        return []
    messages = messages_from_dict(conversation.transcript)
    return trim_messages(
        messages,
        max_tokens=MAX_HISTORY_MESSAGES,
        token_counter=len,  # count messages, not tokens - simple and predictable
        strategy="last",
        start_on="human",
        include_system=False,  # the graph supplies its own preamble every turn
        allow_partial=False,
    )


@transaction.atomic
def submit_question(conversation: Conversation, question: str) -> ChatMessage:
    """Run one agent turn and persist everything it produced.

    Returns the assistant `ChatMessage` (with `.artifacts` populated).
    """
    ChatMessage.objects.create(
        conversation=conversation, role="user", content=question
    )

    session = get_session(conversation)

    try:
        reply = run_agent(
            session,
            question,
            history=_history(conversation),
            model=getattr(settings, "ANALYST_MODEL", None),
            api_key=getattr(settings, "OPENAI_API_KEY", None),
        )
    except Exception as exc:  # noqa: BLE001 - never 500 on a model/API failure
        log.exception("Analyst turn failed for conversation %s", conversation.id)
        return ChatMessage.objects.create(
            conversation=conversation,
            role="error",
            content=(
                "The analysis could not be completed. "
                f"{type(exc).__name__}: {exc}"
            ),
        )

    message = ChatMessage.objects.create(
        conversation=conversation,
        role="assistant",
        content=reply.text,
        tool_calls=_summarise_tool_calls(reply.tool_calls),
    )

    for item in reply.artifacts:
        path = Path(session.artifact_dir) / item["filename"]
        if not path.exists():
            continue
        artifact = Artifact(
            conversation=conversation,
            message=message,
            kind=item["kind"],
            title=item["title"],
        )
        with path.open("rb") as handle:
            artifact.file.save(path.name, File(handle), save=True)

    conversation.transcript = messages_to_dict(reply.messages)
    if not conversation.title:
        conversation.title = question[:80]
    conversation.save(update_fields=["transcript", "title", "updated_at"])

    return message


def _summarise_tool_calls(calls: list[dict]) -> list[dict]:
    """Trim tool-call args to something a details panel can show."""
    out = []
    for call in calls:
        args = call.get("args", {})
        preview = args.get("code") or args.get("markdown") or ""
        out.append(
            {
                "name": call.get("name", ""),
                "preview": preview[:1500],
                "title": args.get("title", "") or args.get("variable", ""),
            }
        )
    return out
