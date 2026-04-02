"""Sub-agent spawning — delegate contained tasks to a fast local model."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import Agent, Tool
from pydantic_ai.messages import ModelMessage

from forge.agent.deps import AgentDeps
from forge.agent.hooks import HookRegistry
from forge.agent.permissions import PermissionPolicy
from forge.agent.tools import (
    ALL_TOOLS,
    edit_file,
    list_files,
    read_file,
    run_command,
    search_code,
    write_file,
)
from forge.agent.worktree import (
    WorktreeInfo,
    create_worktree,
    is_git_repo,
    remove_worktree,
)
from forge.config import settings
from forge.log import get_logger

logger = get_logger(__name__)

# Sub-agent gets a focused toolset (no web, no memory, no tasks)
SUBAGENT_TOOLS: list[Tool] = [
    Tool(read_file),
    Tool(search_code),
    Tool(list_files),
    Tool(write_file, sequential=True),
    Tool(edit_file, sequential=True),
    Tool(run_command, sequential=True),
]

SUBAGENT_SYSTEM = """\
You are a focused coding sub-agent. You have been delegated a specific task.

Rules:
- Complete the task described in the prompt. Do not deviate.
- Work only within the provided working directory.
- When done, respond with a clear summary of what you changed and any issues encountered.
- Do not ask questions — make reasonable decisions and document them.
- If you cannot complete the task, explain why clearly.
"""


@dataclass
class SubagentResult:
    """Result from a sub-agent execution."""

    output: str
    worktree: WorktreeInfo | None
    messages: list[ModelMessage]
    success: bool


async def run_subagent(
    task: str,
    cwd: Path,
    *,
    model: str | None = None,
    isolate: bool = True,
    timeout: float = 300.0,
    system: str = SUBAGENT_SYSTEM,
) -> SubagentResult:
    """Spawn a sub-agent to handle a contained task.

    Args:
        task: The task description/prompt for the sub-agent.
        cwd: Working directory for the sub-agent.
        model: Model to use (defaults to fast model).
        isolate: If True and in a git repo, create an isolated worktree.
        timeout: Maximum execution time in seconds.
        system: System prompt override.

    Returns:
        SubagentResult with output, worktree info, and message history.
    """
    from forge.core.project import build_project_context

    # Ensure OLLAMA_BASE_URL is set
    import os

    if "OLLAMA_BASE_URL" not in os.environ:
        base = settings.ollama.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        os.environ["OLLAMA_BASE_URL"] = base

    model_name = model or settings.agent.delegate_model or settings.ollama.heavy_model
    worktree_info: WorktreeInfo | None = None
    work_dir = cwd

    # Create isolated worktree if requested and possible
    if isolate and is_git_repo(cwd):
        try:
            worktree_info = create_worktree(cwd)
            work_dir = worktree_info.path
            logger.debug("Sub-agent worktree: %s", work_dir)
        except RuntimeError:
            logger.debug("Worktree creation failed, running in-place", exc_info=True)

    # Build system prompt with project context
    full_system = system
    project_ctx = build_project_context(work_dir)
    if project_ctx:
        full_system += f"\n\n{project_ctx}"

    # Create sub-agent with limited tools and permissive policy
    from forge.models.ollama import _model_settings
    agent: Agent[AgentDeps, str] = Agent(
        model=f"ollama:{model_name}",
        instructions=full_system,
        tools=SUBAGENT_TOOLS,
        deps_type=AgentDeps,
        model_settings=_model_settings(timeout=int(timeout), num_ctx=min(settings.agent.num_ctx, 32768)),
        retries=2,
    )

    from rich.console import Console

    deps = AgentDeps(
        cwd=work_dir,
        console=Console(file=None, force_terminal=False, no_color=True),
        permission=PermissionPolicy.YOLO,  # Sub-agents run autonomously
        worktree=worktree_info,
        hook_registry=HookRegistry(),
    )

    try:
        result = await asyncio.wait_for(
            agent.run(task, deps=deps),
            timeout=timeout,
        )
        return SubagentResult(
            output=result.output,
            worktree=worktree_info,
            messages=result.all_messages(),
            success=True,
        )
    except asyncio.TimeoutError:
        return SubagentResult(
            output=f"Sub-agent timed out after {timeout}s",
            worktree=worktree_info,
            messages=[],
            success=False,
        )
    except Exception as e:
        return SubagentResult(
            output=f"Sub-agent error: {e}",
            worktree=worktree_info,
            messages=[],
            success=False,
        )


async def run_subagent_and_merge(
    task: str,
    cwd: Path,
    *,
    model: str | None = None,
    timeout: float = 300.0,
) -> SubagentResult:
    """Run a sub-agent in a worktree and offer to merge changes back.

    This is the high-level API used by the delegate tool. The worktree
    is always cleaned up — if the sub-agent made changes, they remain
    on the worktree branch for the main agent to merge.
    """
    result = await run_subagent(
        task, cwd, model=model, isolate=True, timeout=timeout
    )

    if result.worktree and result.success:
        # Check if the sub-agent made any changes
        from forge.agent.worktree import _run_git

        diff = _run_git(["diff", "--stat", "HEAD"], result.worktree.path)
        staged = _run_git(["diff", "--cached", "--stat"], result.worktree.path)
        has_changes = bool(diff.stdout.strip() or staged.stdout.strip())

        if has_changes:
            result.output += (
                f"\n\nChanges on branch `{result.worktree.branch}`. "
                f"To merge: `git merge {result.worktree.branch}`"
            )
        else:
            # No changes — clean up the worktree
            try:
                remove_worktree(result.worktree)
                result.worktree = None
            except Exception:
                logger.debug("Worktree cleanup failed", exc_info=True)

    return result
