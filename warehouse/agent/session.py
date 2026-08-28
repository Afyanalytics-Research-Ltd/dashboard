"""
An `AnalysisSession` is everything one conversation needs: the loaded
DataFrames, the live execution namespace (so variables persist between the
agent's turns, exactly like a notebook kernel), and the artifacts it produced.

Tools are bound to a session with a closure rather than read from a global,
so two users analysing two workbooks can never see each other's frames.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .workbook import build_namespace, load_workbook, suggest_joins, workbook_overview

ArtifactKind = Literal["chart", "table", "report"]


@dataclass(slots=True)
class Artifact:
    """A file the agent produced. `path` is on disk; `url` is filled in by the
    Django layer once it knows MEDIA_URL."""

    kind: ArtifactKind
    title: str
    path: Path
    url: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "title": self.title,
            "filename": self.path.name,
            "url": self.url,
        }


@dataclass
class AnalysisSession:
    """One workbook + one persistent kernel + the artifacts produced from it."""

    source_path: Path
    artifact_dir: Path
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    sheet_to_var: dict[str, str] = field(default_factory=dict)
    namespace: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    exec_timeout: float = 30.0

    @classmethod
    def open(
        cls,
        source_path: str | Path,
        artifact_dir: str | Path,
        *,
        exec_timeout: float = 30.0,
    ) -> "AnalysisSession":
        source_path = Path(source_path)
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        frames, sheet_to_var = load_workbook(source_path)
        session = cls(
            source_path=source_path,
            artifact_dir=artifact_dir,
            frames=frames,
            sheet_to_var=sheet_to_var,
            exec_timeout=exec_timeout,
        )
        session.namespace = build_namespace(frames)
        return session

    # -- context the model is primed with ---------------------------------- #

    def overview(self) -> str:
        parts = [
            f"Workbook: `{self.source_path.name}`",
            "",
            workbook_overview(self.frames, self.sheet_to_var),
        ]
        joins = suggest_joins(self.frames)
        if joins:
            parts += [joins, ""]
        return "\n".join(parts)

    # -- artifact helpers --------------------------------------------------- #

    def new_artifact_path(self, stem: str, suffix: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:60] or "output"
        return self.artifact_dir / f"{safe}-{uuid.uuid4().hex[:8]}{suffix}"

    def record(self, kind: ArtifactKind, title: str, path: Path) -> Artifact:
        artifact = Artifact(kind=kind, title=title, path=path)
        self.artifacts.append(artifact)
        return artifact
