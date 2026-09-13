from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceContext:
    """Immutable project identity carried into every run and child thread."""

    root: Path
    project_id: str

    @classmethod
    def from_path(cls, path: Path | str) -> WorkspaceContext:
        root = Path(path).expanduser().resolve()
        identity = os.path.normcase(str(root)).encode("utf-8")
        return cls(root, hashlib.sha256(identity).hexdigest()[:24])


_workspace_context: ContextVar[WorkspaceContext | None] = ContextVar(
    "workspace_context", default=None
)


def active_workspace() -> WorkspaceContext | None:
    return _workspace_context.get()


@contextmanager
def workspace_context(workspace: WorkspaceContext):
    token = _workspace_context.set(workspace)
    try:
        yield workspace
    finally:
        _workspace_context.reset(token)
