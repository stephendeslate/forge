"""Shared dependencies injected into agent tools via RunContext."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from forge.agent.permissions import PermissionPolicy

if TYPE_CHECKING:
    from forge.agent.circuit_breaker import ToolCallTracker
    from forge.agent.escalation import ModelEscalator
    from forge.agent.hooks import HookRegistry
    from forge.agent.status import StatusTracker
    from forge.agent.task_store import TaskStore
    from forge.agent.turn_buffer import TurnBuffer
    from forge.agent.worktree import WorktreeInfo
    from forge.models.ollama import OllamaMonitor
    from forge.storage.database import Database


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
    rag_db: Database | None = None
    rag_project: str | None = None
    # Plan tracking
    active_plan: str | None = None
    # Worktree isolation
    worktree: WorktreeInfo | None = None
    # Hooks
    hook_registry: HookRegistry | None = None
    # Memory
    memory_db: Database | None = None
    memory_project: str | None = None
    # Task tracking
    task_store: TaskStore | None = None
    # Ollama model lifecycle
    ollama_monitor: OllamaMonitor | None = None
    # Token tracking (real counts from Ollama API)
    tokens_in: int = 0
    tokens_out: int = 0
    # Circuit breaker
    circuit_breaker: ToolCallTracker | None = None
    # Model escalation
    escalator: ModelEscalator | None = None
    # MCP servers (for passing to sub-agents)
    mcp_servers: list | None = None
    # Syntax/lint feedback injection
    _post_tool_feedback: str | None = None
    lint_results: str | None = None
