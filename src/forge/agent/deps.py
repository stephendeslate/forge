"""Shared dependencies injected into agent tools via RunContext."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from forge.agent.permissions import PermissionPolicy


@dataclass
class AgentDeps:
    """State shared across all tool invocations in an agent run."""

    cwd: Path
    console: Console = field(default_factory=Console)
    permission: PermissionPolicy = PermissionPolicy.AUTO
