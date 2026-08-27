"""A pandas data-analyst agent built on LangGraph + LangChain.

Typical use outside Django::

    from agent import AnalysisSession, ask

    session = AnalysisSession.open("sales.xlsx", artifact_dir="./out")
    reply = ask(session, "Which region grew fastest last quarter?")
    print(reply.text)
    for artifact in reply.artifacts:
        print(artifact["kind"], artifact["filename"])
"""

from .graph import AnalystReply, ask, build_graph
from .sandbox import ExecResult, UnsafeCodeError, execute, validate_source
from .session import AnalysisSession, Artifact
from .workbook import load_workbook, profile_frame, workbook_overview

__all__ = [
    "AnalysisSession",
    "AnalystReply",
    "Artifact",
    "ExecResult",
    "UnsafeCodeError",
    "ask",
    "build_graph",
    "execute",
    "load_workbook",
    "profile_frame",
    "validate_source",
    "workbook_overview",
]
