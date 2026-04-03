"""Session lifecycle — setup and cleanup extracted from agent_repl()."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.messages import ModelMessage
from rich.console import Console
from rich.panel import Panel

from forge import __version__
from forge.agent.deps import AgentDeps
from forge.agent.hooks import (
    HookRegistry,
    PostToolUse,
    TurnEnd,
)
from forge.agent.permissions import PermissionPolicy
from forge.agent.task_store import TaskStore
from forge.agent.tools import ALL_TOOLS, DELEGATE_TOOLS, MEMORY_TOOLS, TASK_TOOLS
from forge.config import settings
from forge.log import get_logger

if TYPE_CHECKING:
    from contextlib import AsyncExitStack

    from forge.agent.worktree import WorktreeInfo
    from forge.storage.database import Database

logger = get_logger(__name__)


async def setup_model_monitor(deps: AgentDeps, console: Console) -> None:
    """Create OllamaMonitor, optionally preload the heavy model."""
    from forge.models.ollama import OllamaMonitor

    monitor = OllamaMonitor()
    deps.ollama_monitor = monitor

    if not await monitor.health_check():
        logger.warning("Ollama not responsive — monitor disabled")
        deps.ollama_monitor = None
        return

    if settings.agent.preload_model:
        model = settings.ollama.heavy_model
        if not await monitor.is_loaded(model):
            console.print(f"[dim]Preloading {model}...[/dim]")
            ok = await monitor.preload(model, num_ctx=settings.agent.num_ctx)
            if ok:
                console.print(f"[dim]Model {model} loaded.[/dim]")
            else:
                console.print("[yellow]Model preload timed out — will load on first request.[/yellow]")


def setup_worktree(
    cwd: Path, worktree_name: str | None, console: Console,
) -> tuple[Path, WorktreeInfo | None]:
    """Create a git worktree if requested. Returns (cwd, worktree_info)."""
    if worktree_name is None:
        return cwd, None

    from forge.agent.worktree import create_worktree, is_git_repo

    if not is_git_repo(cwd):
        console.print("[red]Not a git repository — cannot create worktree.[/red]")
        raise SystemExit(1)
    try:
        wt_info = create_worktree(cwd, worktree_name or None)
        console.print(
            f"[green]Worktree created:[/green] {wt_info.path}\n"
            f"[green]Branch:[/green] {wt_info.branch}"
        )
        return wt_info.path, wt_info
    except RuntimeError as e:
        console.print(f"[red]Worktree error:[/red] {e}")
        raise SystemExit(1) from e


async def setup_persistence(
    resume_id: str | None,
    deps: AgentDeps,
    console: Console,
    system: str,
) -> tuple[Database | None, str, list[ModelMessage] | None]:
    """Connect to DB, create or resume session.

    Returns (db, session_id, message_history).
    """
    from forge.agent.loop import _connect_db, _load_agent_history

    db = await _connect_db()
    session_id = resume_id or str(uuid.uuid4())
    message_history: list[ModelMessage] | None = None

    if db and resume_id:
        loaded = await _load_agent_history(db, session_id)
        if loaded:
            message_history = loaded
            console.print(f"[dim]Resumed session with {len(loaded)} messages.[/dim]")
        # Resume task store
        try:
            task_json = await db.load_task_store(session_id)
            if task_json:
                deps.task_store = TaskStore.from_json(task_json)
                open_tasks = deps.task_store.list_open()
                if open_tasks:
                    console.print(f"[dim]Restored {len(open_tasks)} open task(s).[/dim]")
        except Exception:
            logger.debug("Task store load failed", exc_info=True)
    elif db:
        # Detect mode from system prompt identity check
        from forge.agent.loop import AGENT_SYSTEM

        try:
            await db.create_session(
                session_id, mode="agent" if system is AGENT_SYSTEM else "chat",
            )
        except Exception:
            logger.warning("Session creation failed, disabling persistence", exc_info=True)
            db = None

    return db, session_id, message_history


async def setup_memory(db: Database | None, deps: AgentDeps, cwd: Path) -> int:
    """Wire memory DB into deps. Returns memory count."""
    if not db:
        return 0
    deps.memory_db = db
    deps.memory_project = cwd.name
    try:
        return await db.count_memories(cwd.name)
    except Exception:
        logger.debug("Memory count failed", exc_info=True)
        return 0


async def setup_rag(
    db: Database | None,
    deps: AgentDeps,
    cwd: Path,
    console: Console,
    worktree_info: Any | None,
) -> tuple[bool, str]:
    """Check RAG availability and auto-reindex stale files.

    Returns (rag_available, rag_project_name).
    """
    rag_project_name = worktree_info.base_dir.name if worktree_info else cwd.name
    rag_available = False

    if not db:
        return rag_available, rag_project_name

    try:
        stats = await db.get_project_stats(rag_project_name)
        if stats.get("chunk_count", 0) > 0:
            deps.rag_db = db
            deps.rag_project = rag_project_name
            rag_available = True
    except Exception:
        logger.debug("RAG stats check failed", exc_info=True)

    # Auto-reindex stale files in background
    if rag_available:
        try:
            from forge.rag.indexer import find_stale_files, reindex_files

            stale = await find_stale_files(cwd, db, rag_project_name)
            if stale:
                console.print(f"[dim]Auto-reindexing {len(stale)} changed file(s)...[/dim]")
                asyncio.create_task(reindex_files(stale, cwd, db, rag_project_name))
        except Exception:
            logger.debug("Auto-reindex check failed", exc_info=True)

    return rag_available, rag_project_name


async def setup_mcp(cwd: Path) -> tuple[list, AsyncExitStack]:
    """Discover and start MCP servers. Returns (servers, exit_stack)."""
    from contextlib import AsyncExitStack

    from forge.agent.mcp_config import load_all_mcp_servers

    mcp_servers = load_all_mcp_servers(cwd)
    mcp_stack = AsyncExitStack()
    mcp_started: list = []

    for server in mcp_servers:
        try:
            await mcp_stack.enter_async_context(server)
            mcp_started.append(server)
        except Exception:
            sid = getattr(server, "id", server)
            logger.warning("Failed to start MCP server: %s", sid, exc_info=True)

    return mcp_started, mcp_stack


def build_agent_with_tools(
    system: str,
    cwd: Path,
    deps: AgentDeps,
    rag_available: bool,
    mcp_servers: list,
) -> tuple[Agent, list[Tool], str]:
    """Build the agent with conditional tools. Returns (agent, extra_tools, system)."""
    from forge.agent.loop import create_agent

    extra_tools: list[Tool] = list(TASK_TOOLS)
    extra_tools.extend(DELEGATE_TOOLS)

    if deps.memory_db:
        extra_tools.extend(MEMORY_TOOLS)

    if rag_available:
        from forge.agent.tools import rag_search

        extra_tools.append(rag_search)
        system = (
            system + "\n\nUse rag_search for semantic code search (conceptual queries). "
            "Use search_code for exact text/regex matches."
        )

    if mcp_servers:
        system += (
            "\n\n## MCP tools\n"
            "Some tools come from external MCP servers. They are prefixed with the server name "
            "(e.g. \"filesystem_read_file\"). Use them like any other tool. "
            "If an MCP tool fails, report the error to the user."
        )

    # Store MCP servers in deps so delegate tool can pass them to sub-agents
    deps.mcp_servers = mcp_servers or None

    agent = create_agent(
        system=system, cwd=cwd, tools=ALL_TOOLS + extra_tools,
        toolsets=mcp_servers or None,
    )
    return agent, extra_tools, system


def wire_dynamic_prompts(agent: Agent, deps: AgentDeps) -> None:
    """Register dynamic system prompt injectors for tasks and memory."""

    @agent.system_prompt
    async def _inject_tasks(ctx: RunContext[AgentDeps]) -> str:
        if ctx.deps.task_store:
            return ctx.deps.task_store.to_prompt()
        return ""

    @agent.system_prompt
    async def _inject_lint(ctx: RunContext[AgentDeps]) -> str:
        if ctx.deps.lint_results:
            results = ctx.deps.lint_results
            ctx.deps.lint_results = None  # Clear after injection
            return f"## Lint issues from previous turn\n```\n{results}\n```\nFix these before proceeding."
        return ""

    @agent.system_prompt
    async def _inject_memories(ctx: RunContext[AgentDeps]) -> str:
        if ctx.deps.memory_db and ctx.deps.memory_project:
            try:
                from forge.agent.memory import get_startup_memories

                rows = await get_startup_memories(
                    ctx.deps.memory_db, ctx.deps.memory_project, limit=10,
                )
                if rows:
                    lines = ["## Remembered context\n"]
                    for r in rows:
                        lines.append(f"- [{r.category}] **{r.subject}**: {r.content}")
                    return "\n".join(lines)
            except Exception:
                logger.debug("Memory injection failed", exc_info=True)
        return ""


def wire_lint_hooks(hook_registry: HookRegistry, deps: AgentDeps) -> None:
    """Register turn-end background lint hook.

    Tracks files modified during a turn via PostToolUse, then at TurnEnd
    runs a fast lint pass (ruff check) in the background. Results are
    stored on deps for injection into the next turn's context.
    """
    _pending_lint: set[Path] = set()

    async def _track_writes(event: PostToolUse) -> None:
        if event.tool_name in ("write_file", "edit_file"):
            file_path = event.args.get("file_path", "")
            if file_path.endswith(".py"):
                p = Path(file_path)
                if not p.is_absolute():
                    p = (deps.cwd / p).resolve()
                _pending_lint.add(p)

    async def _run_lint(event: TurnEnd) -> None:
        if not _pending_lint:
            return
        files = [str(p) for p in _pending_lint if p.exists()]
        _pending_lint.clear()
        if not files:
            return

        import subprocess

        try:
            result = subprocess.run(
                ["ruff", "check", "--no-fix", "--output-format=concise", *files],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(deps.cwd),
            )
            if result.stdout.strip():
                deps.lint_results = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # ruff not installed or timed out — skip silently

    hook_registry.on(PostToolUse, _track_writes, priority=15)
    hook_registry.on(TurnEnd, _run_lint, priority=20)


def wire_syntax_hooks(hook_registry: HookRegistry, deps: AgentDeps) -> None:
    """Register post-write syntax check hooks.

    After write_file/edit_file on Python files, runs py_compile to catch
    syntax errors immediately. Injects error feedback into tool result via deps.
    """

    async def _check_syntax(event: PostToolUse) -> None:
        if event.tool_name not in ("write_file", "edit_file"):
            return
        file_path = event.args.get("file_path", "")
        if not file_path.endswith(".py"):
            return

        p = Path(file_path)
        if not p.is_absolute():
            p = (deps.cwd / p).resolve()

        if not p.exists():
            return

        import subprocess

        try:
            result = subprocess.run(
                ["python", "-m", "py_compile", str(p)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip()
                deps._post_tool_feedback = (
                    f"⚠ SYNTAX ERROR in {p.name}: {error_msg}\n"
                    "Fix this before proceeding."
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # Don't block on check failures

    hook_registry.on(PostToolUse, _check_syntax, priority=10)


def wire_rag_hooks(
    hook_registry: HookRegistry,
    deps: AgentDeps,
    db: Database,
    rag_project_name: str,
) -> set[Path]:
    """Register post-write reindex hooks. Returns the pending-reindex set."""
    _pending_reindex: set[Path] = set()

    async def _reindex_on_write(event: PostToolUse) -> None:
        if event.tool_name in ("write_file", "edit_file"):
            file_path = event.args.get("file_path")
            if file_path:
                p = Path(file_path)
                if not p.is_absolute():
                    p = (deps.cwd / p).resolve()
                else:
                    p = p.resolve()
                _pending_reindex.add(p)

    async def _flush_reindex(event: TurnEnd) -> None:
        if _pending_reindex and db:
            from forge.rag.indexer import reindex_files

            files = list(_pending_reindex)
            _pending_reindex.clear()
            asyncio.create_task(reindex_files(files, deps.cwd, db, rag_project_name))

    hook_registry.on(PostToolUse, _reindex_on_write)
    hook_registry.on(TurnEnd, _flush_reindex)
    return _pending_reindex


def print_welcome(
    console: Console,
    deps: AgentDeps,
    session_id: str,
    db: Database | None,
    system: str,
    rag_available: bool,
    memory_count: int,
    mcp_servers: list,
) -> None:
    """Print the welcome banner."""
    from forge.agent.loop import AGENT_SYSTEM
    from forge.core.project import INSTRUCTION_FILES, detect_project_type

    project_type = detect_project_type(deps.cwd)
    project_info = f"Project: [cyan]{project_type}[/cyan] | " if project_type else ""
    instructions_loaded = any((deps.cwd / f).is_file() for f in INSTRUCTION_FILES)
    instr_info = "[green]instructions loaded[/green]" if instructions_loaded else ""
    rag_info = " | [green]RAG indexed[/green]" if rag_available else ""
    memory_info = f" | [green]{memory_count} memories[/green]" if memory_count else ""
    mcp_info = f" | [green]{len(mcp_servers)} MCP server(s)[/green]" if mcp_servers else ""

    persist_info = f"\nSession: [dim]{session_id[:8]}…[/dim]" if db else ""
    worktree_banner = ""
    if deps.worktree:
        worktree_banner = (
            f"\nWorktree: [cyan]{deps.worktree.branch}[/cyan] "
            f"at [dim]{deps.worktree.path}[/dim]"
        )

    active_model = deps.model_override or settings.ollama.heavy_model
    is_agent = system is AGENT_SYSTEM
    console.print(
        Panel(
            f"[bold]Forge v{__version__}[/bold] — {'agentic coding mode' if is_agent else 'chat + tools'}\n"
            f"Model: [green]{active_model}[/green]\n"
            f"Permissions: [{'green' if deps.permission == PermissionPolicy.YOLO else 'yellow'}]"
            f"{deps.permission.value}[/{'green' if deps.permission == PermissionPolicy.YOLO else 'yellow'}]\n"
            f"{project_info}{instr_info}{rag_info}{memory_info}{mcp_info}\n"
            f"Working directory: [dim]{deps.cwd}[/dim]"
            f"{worktree_banner}"
            f"{persist_info}\n"
            "Type [bold]/help[/bold] for commands, [bold]Ctrl-O[/bold] status, [bold]Ctrl-R[/bold] tools, [bold]Ctrl-D[/bold] exit",
            title="forge agent" if is_agent else "forge",
            border_style="magenta",
        )
    )


async def cleanup(
    deps: AgentDeps,
    hook_registry: HookRegistry,
    mcp_stack: AsyncExitStack,
    db: Database | None,
    session_id: str,
    message_history: list[ModelMessage] | None,
    console: Console,
) -> None:
    """End-of-session cleanup: emit events, save state, close resources."""
    from forge.agent.hooks import SessionEnd
    from forge.agent.loop import _save_agent_session

    # Emit SessionEnd
    msg_count = len(message_history) if message_history else 0
    await hook_registry.emit(SessionEnd(session_id=session_id, message_count=msg_count))

    # Close Ollama monitor
    if deps.ollama_monitor:
        await deps.ollama_monitor.close()

    # Shut down MCP servers
    await mcp_stack.aclose()

    # Persist final state
    if db and message_history:
        await _save_agent_session(db, session_id, message_history)
    if db and deps.task_store:
        try:
            await db.save_task_store(session_id, deps.task_store.to_json())
        except Exception:
            logger.debug("Final task store save failed", exc_info=True)
    if db:
        await db.close()

    # Worktree cleanup
    if deps.worktree:
        from forge.agent.worktree import prompt_worktree_cleanup, remove_worktree

        try:
            keep = await prompt_worktree_cleanup(deps.worktree)
            if keep:
                deps.worktree.unregister_atexit()
                console.print(
                    f"[dim]Keeping worktree at {deps.worktree.path} "
                    f"(branch: {deps.worktree.branch})[/dim]"
                )
            else:
                remove_worktree(deps.worktree)
                console.print("[dim]Worktree removed.[/dim]")
        except Exception:
            logger.debug("Worktree cleanup failed", exc_info=True)
