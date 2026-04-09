"""Shared dependencies injected into agent tools via RunContext."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from forge.agent.permissions import PermissionPolicy

if TYPE_CHECKING:
    from pydantic_ai.models import Model

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
    """State shared across all tool invocations in an agent run.

    Fields are grouped into two categories:

    **Session-scoped** (cwd, console, permission, etc.) — set once at session
    start and stable across turns.

    **Turn-scoped** (prefixed with _ or grouped below the marker comment) —
    mutable state that should be reset between turns via ``reset_turn()``.
    """

    # --- Session-scoped dependencies ---
    cwd: Path
    console: Console = field(default_factory=Console)
    permission: PermissionPolicy = PermissionPolicy.AUTO
    status_tracker: StatusTracker | None = None
    thinking_enabled: bool = False
    status_visible: bool = True
    tools_visible: bool = True
    turn_buffer: TurnBuffer | None = None
    url_cache: dict[str, str] = field(default_factory=dict)
    # Model switching (str for Ollama/qualified names, Model for pydantic-ai Model objects)
    model_override: str | Model | None = None
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
    # Cloud reasoning (Gemini)
    cloud_reasoning_enabled: bool = False
    # Background processes (auto-backgrounded long commands)
    _background_procs: dict[int, tuple[asyncio.Task, asyncio.subprocess.Process]] = field(default_factory=dict)
    # Conversation summary for sub-agent context injection
    _conversation_summary: str = ""

    # --- Turn-scoped mutable state (reset via reset_turn()) ---
    _post_tool_feedback: list[str] = field(default_factory=list)
    lint_results: str | None = None
    test_results: str | None = None
    test_command: str | None = None
    _test_command_searched: bool = False
    _files_modified_this_turn: list[str] = field(default_factory=list)
    critique_results: str | None = None
    _cloud_recovery_pending: bool = False
    _active_exemplar_ids: list[int] = field(default_factory=list)
    _exemplar_context: str | None = None  # single-shot system prompt injection
    _write_escalated: bool = False
    _last_prompt_eval_count: int = 0
    _memory_cache_dirty: bool = False

    def reset_turn(self) -> None:
        """Reset all turn-scoped mutable state for a new turn."""
        self._post_tool_feedback = []
        self.lint_results = None
        self.test_results = None
        # Note: test_command persists across turns (cached detection result)
        self._files_modified_this_turn = []
        self.critique_results = None
        self._cloud_recovery_pending = False
        self._active_exemplar_ids = []
        self._exemplar_context = None
        self._write_escalated = False
        self._memory_cache_dirty = False

    async def cleanup(self) -> None:
        """Clean up orphan background processes and resources."""
        for pid, (task, proc) in list(self._background_procs.items()):
            try:
                if proc.returncode is None:
                    import os
                    import signal
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except (ProcessLookupError, OSError):
                        proc.kill()
                task.cancel()
            except Exception:
                pass
        self._background_procs.clear()
