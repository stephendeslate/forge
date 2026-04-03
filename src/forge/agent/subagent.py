"""Sub-agent spawning — delegate contained tasks to a fast local model."""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import Agent, Tool
from pydantic_ai.messages import ModelMessage

from forge.agent.deps import AgentDeps
from forge.agent.hooks import HookRegistry, PostToolUse
from forge.agent.permissions import PermissionPolicy
from forge.agent.tools import (
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

# Regex patterns for detecting genuine task-level failure (not domain references)
FAILURE_PATTERNS = [
    r"(?i)^(?:error|fatal|exception):",                       # Line starts with "error:", "fatal:", etc.
    r"(?i)\bcould not (?:complete|finish|accomplish)\b",      # Explicit task failure
    r"(?i)\bunable to (?:complete|finish|accomplish)\b",      # Explicit task failure
    r"(?i)\bfailed to (?:complete|finish|accomplish)\b",      # Explicit task failure
    r"(?i)\btimed? ?out\b.*\b(?:after|waiting)\b",           # Timeout with duration
    r"(?i)^Traceback \(most recent call last\)",              # Python traceback
]


@dataclass
class SubagentResult:
    """Result from a sub-agent execution."""

    output: str
    worktree: WorktreeInfo | None
    messages: list[ModelMessage]
    success: bool


@dataclass
class MergeResult:
    """Result from attempting to merge a worktree branch."""

    merged: bool
    conflict: bool
    message: str


def _validate_output(result: SubagentResult) -> SubagentResult:
    """Validate sub-agent output using structured pattern matching.

    Uses regex patterns that detect task-level failure ("could not complete")
    rather than domain references ("error handling was improved").
    Sets result.success = False if problems are detected.
    """
    import re

    if not result.output or not result.output.strip():
        result.success = False
        result.output = result.output or "(empty output from sub-agent)"
        return result

    for line in result.output.strip().split("\n"):
        stripped = line.strip()
        for pattern in FAILURE_PATTERNS:
            if re.search(pattern, stripped):
                result.success = False
                return result

    return result


def _commit_pending(worktree: WorktreeInfo) -> bool:
    """Stage all changes and commit in the worktree. Returns True if committed."""
    from forge.agent.worktree import _run_git

    # Stage everything
    _run_git(["add", "-A"], worktree.path)

    # Check if there's anything to commit
    status = _run_git(["status", "--porcelain"], worktree.path)
    if not status.stdout.strip():
        return False

    _run_git(
        ["commit", "-m", f"subagent: changes on {worktree.branch}"],
        worktree.path,
    )
    return True


def _generate_diff_summary(worktree: WorktreeInfo) -> str:
    """Generate a structured diff summary of the worktree's latest commit."""
    from forge.agent.worktree import _run_git

    stat = _run_git(["diff", "--stat", "HEAD~1"], worktree.path)
    shortstat = _run_git(["diff", "--shortstat", "HEAD~1"], worktree.path)

    parts = []
    if stat.stdout.strip():
        parts.append(stat.stdout.strip())
    if shortstat.stdout.strip():
        parts.append(shortstat.stdout.strip())

    return "\n".join(parts) if parts else "(no diff available)"


def _auto_merge(worktree: WorktreeInfo, base_dir: Path) -> MergeResult:
    """Merge the worktree branch back into the base repo.

    Returns MergeResult indicating success, conflict, or failure.
    """
    from forge.agent.worktree import _run_git

    # Check base repo working tree is clean
    status = _run_git(["status", "--porcelain"], base_dir)
    if status.stdout.strip():
        return MergeResult(
            merged=False,
            conflict=False,
            message="Base repo has uncommitted changes — skipping auto-merge",
        )

    # Attempt merge
    try:
        result = subprocess.run(
            ["git", "merge", "--no-ff", worktree.branch, "-m",
             f"Merge sub-agent branch '{worktree.branch}'"],
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return MergeResult(merged=False, conflict=False, message="Merge timed out")

    if result.returncode == 0:
        # Success — clean up worktree
        try:
            remove_worktree(worktree)
        except Exception:
            logger.debug("Post-merge worktree cleanup failed", exc_info=True)
        return MergeResult(merged=True, conflict=False, message="Merged successfully")

    # Merge failed — likely conflict
    if "CONFLICT" in result.stdout or "conflict" in result.stderr.lower():
        # Abort the conflicted merge
        subprocess.run(
            ["git", "merge", "--abort"],
            cwd=base_dir,
            capture_output=True,
            timeout=10,
        )
        return MergeResult(
            merged=False,
            conflict=True,
            message=f"Merge conflict — branch `{worktree.branch}` preserved for manual resolution",
        )

    return MergeResult(
        merged=False,
        conflict=False,
        message=f"Merge failed: {result.stderr.strip() or result.stdout.strip()}",
    )


async def run_subagent(
    task: str,
    cwd: Path,
    *,
    model: str | None = None,
    isolate: bool = True,
    timeout: float = 300.0,
    system: str = SUBAGENT_SYSTEM,
    parent_hooks: HookRegistry | None = None,
    mcp_servers: list | None = None,
) -> SubagentResult:
    """Spawn a sub-agent to handle a contained task.

    Args:
        task: The task description/prompt for the sub-agent.
        cwd: Working directory for the sub-agent.
        model: Model to use (defaults to fast model).
        isolate: If True and in a git repo, create an isolated worktree.
        timeout: Maximum execution time in seconds.
        system: System prompt override.
        parent_hooks: If provided, PostToolUse handlers are inherited for observability.

    Returns:
        SubagentResult with output, worktree info, and message history.
    """
    # Ensure OLLAMA_BASE_URL is set
    import os

    from forge.core.project import build_project_context

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
        toolsets=mcp_servers or [],
        deps_type=AgentDeps,
        model_settings=_model_settings(timeout=int(timeout), num_ctx=min(settings.agent.num_ctx, 32768)),
        retries=2,
    )

    from rich.console import Console

    # Build hook registry — inherit PostToolUse handlers from parent for observability
    sub_hooks = HookRegistry()
    if parent_hooks:
        for priority, handler in parent_hooks.get_handlers(PostToolUse):
            sub_hooks.on(PostToolUse, handler, priority=priority)

    deps = AgentDeps(
        cwd=work_dir,
        console=Console(file=None, force_terminal=False, no_color=True),
        permission=PermissionPolicy.YOLO,  # Sub-agents run autonomously
        worktree=worktree_info,
        hook_registry=sub_hooks,
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
    parent_hooks: HookRegistry | None = None,
    auto_merge: bool = True,
    mcp_servers: list | None = None,
) -> SubagentResult:
    """Run a sub-agent in a worktree, validate output, and optionally merge.

    This is the high-level API used by the delegate tool. Flow:
    1. Run sub-agent in isolated worktree
    2. Validate output for error indicators
    3. Commit pending changes in worktree
    4. Generate diff summary
    5. Auto-merge if requested and base repo is clean
    6. Clean up worktree on success or no changes

    Args:
        parent_hooks: Parent hook registry — PostToolUse handlers inherited for observability.
        auto_merge: If True (default), automatically merge the branch back on success.
    """
    result = await run_subagent(
        task, cwd, model=model, isolate=True, timeout=timeout,
        parent_hooks=parent_hooks, mcp_servers=mcp_servers,
    )

    # Validate output for error indicators
    result = _validate_output(result)

    if not result.worktree:
        return result

    if not result.success:
        # Failed — clean up worktree, don't try to merge
        try:
            remove_worktree(result.worktree)
            result.worktree = None
        except Exception:
            logger.debug("Worktree cleanup failed", exc_info=True)
        return result

    # Check if the sub-agent made any changes
    from forge.agent.worktree import _run_git

    diff = _run_git(["diff", "--stat", "HEAD"], result.worktree.path)
    staged = _run_git(["diff", "--cached", "--stat"], result.worktree.path)
    has_changes = bool(diff.stdout.strip() or staged.stdout.strip())

    if not has_changes:
        # No changes — clean up the worktree
        try:
            remove_worktree(result.worktree)
            result.worktree = None
        except Exception:
            logger.debug("Worktree cleanup failed", exc_info=True)
        return result

    # Commit pending changes
    committed = _commit_pending(result.worktree)

    # Generate diff summary
    if committed:
        diff_summary = _generate_diff_summary(result.worktree)
        result.output += f"\n\n**Changes:**\n```\n{diff_summary}\n```"

    # Auto-merge if requested
    if auto_merge:
        merge_result = _auto_merge(result.worktree, cwd)
        if merge_result.merged:
            result.output += f"\n\n{merge_result.message}"
            result.worktree = None  # Cleaned up by _auto_merge
        elif merge_result.conflict:
            result.output += f"\n\n{merge_result.message}"
        else:
            # Non-conflict failure — keep branch info
            result.output += (
                f"\n\n{merge_result.message}\n"
                f"Branch `{result.worktree.branch}` preserved. "
                f"To merge manually: `git merge {result.worktree.branch}`"
            )
    else:
        result.output += (
            f"\n\nChanges on branch `{result.worktree.branch}`. "
            f"To merge: `git merge {result.worktree.branch}`"
        )

    return result


async def run_subagents_parallel(
    tasks: list[str],
    cwd: Path,
    *,
    model: str | None = None,
    timeout: float = 300.0,
    parent_hooks: HookRegistry | None = None,
    mcp_servers: list | None = None,
    max_concurrent: int = 4,
) -> list[SubagentResult]:
    """Run multiple sub-agents concurrently, respecting max_concurrent.

    Each sub-agent gets its own isolated worktree. Results are returned
    in the same order as the input tasks list.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_one(task: str) -> SubagentResult:
        async with semaphore:
            return await run_subagent_and_merge(
                task=task,
                cwd=cwd,
                model=model,
                timeout=timeout,
                parent_hooks=parent_hooks,
                mcp_servers=mcp_servers,
            )

    return list(await asyncio.gather(*[_run_one(t) for t in tasks]))
