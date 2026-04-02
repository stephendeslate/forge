"""Shared dependencies injected into agent tools via RunContext."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from forge.agent.permissions import PermissionPolicy

if TYPE_CHECKING:
    from forge.agent.status import StatusTracker
    from forge.agent.turn_buffer import TurnBuffer


@dataclass
class AgentDeps:
    """State shared across all tool invocations in an agent run."""

    cwd: Path
    console: Console = field(default_factory=Console)
    permission: PermissionPolicy = PermissionPolicy.AUTO
    status_tracker: StatusTracker | None = None
    thinking_enabled: bool = False
    status_visible: bool = True
    tools_visible: bool = True
    turn_buffer: TurnBuffer | None = None
    url_cache: dict[str, str] = field(default_factory=dict)
    # Model switching
    model_override: str | None = None
    # RAG integration
    rag_db: object | None = None  # Database instance when RAG is available
    rag_project: str | None = None
    # Plan tracking
    active_plan: str | None = None
