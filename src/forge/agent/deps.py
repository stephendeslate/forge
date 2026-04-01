"""Shared dependencies injected into agent tools via RunContext."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from forge.agent.permissions import PermissionPolicy

if TYPE_CHECKING:
    from forge.agent.status import StatusTracker


@dataclass
class AgentDeps:
    """State shared across all tool invocations in an agent run."""

    cwd: Path
    console: Console = field(default_factory=Console)
    permission: PermissionPolicy = PermissionPolicy.AUTO
    status_tracker: StatusTracker | None = None
    thinking_enabled: bool = False
    status_visible: bool = True
