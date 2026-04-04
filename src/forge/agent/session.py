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
    TurnStart,
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
    """Register dynamic system prompt injectors for tasks, memory, tests, and critique."""

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
    async def _inject_test_results(ctx: RunContext[AgentDeps]) -> str:
        if ctx.deps.test_results:
            results = ctx.deps.test_results
            ctx.deps.test_results = None  # Clear after injection
            return (
                f"\n\n⚠️ TEST FAILURES from your recent changes:\n```\n{results}\n```\n"
                "Fix these failures before making more changes."
            )
        return ""

    @agent.system_prompt
    async def _inject_critique(ctx: RunContext[AgentDeps]) -> str:
        if ctx.deps.critique_results:
            results = ctx.deps.critique_results
            ctx.deps.critique_results = None  # One-shot injection
            return f"\n\n🔍 AUTO-REVIEW of your recent changes:\n{results}\nAddress these issues."
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


def wire_test_hooks(hook_registry: HookRegistry, deps: AgentDeps) -> None:
    """Register test-driven self-correction hooks.

    Tracks files modified during a turn via PostToolUse. At TurnEnd, auto-detects
    and runs the test command. Failures are stored on deps for injection into the
    next turn's system prompt.
    """

    async def _track_modified_files(event: PostToolUse) -> None:
        if event.tool_name in ("write_file", "edit_file"):
            file_path = event.args.get("file_path", "")
            if file_path:
                p = Path(file_path)
                if not p.is_absolute():
                    p = (deps.cwd / p).resolve()
                deps._files_modified_this_turn.append(str(p))
        elif event.tool_name == "run_command":
            # Detect commands that write files (redirects, tee, etc.)
            cmd = event.args.get("command", "")
            if any(ind in cmd for ind in (">", "tee ", "mv ", "cp ", "sed -i")):
                deps._files_modified_this_turn.append(f"__cmd_write:{cmd[:80]}")

    async def _run_tests(event: TurnEnd) -> None:
        if not settings.agent.test_enabled:
            return
        if len(deps._files_modified_this_turn) < settings.agent.test_min_writes:
            return

        test_cmd = await _detect_test_command(deps)
        if not test_cmd:
            return

        # Scope tests to modified files where possible
        real_files = [
            f for f in deps._files_modified_this_turn
            if not f.startswith("__cmd_write:")
        ]
        scoped_cmd = _scope_test_command(test_cmd, real_files)

        import subprocess

        try:
            result = subprocess.run(
                scoped_cmd,
                capture_output=True,
                text=True,
                timeout=settings.agent.test_timeout,
                cwd=str(deps.cwd),
                shell=True,
            )
            if result.returncode != 0:
                output = (result.stdout + "\n" + result.stderr).strip()
                # Truncate to avoid bloating context
                if len(output) > 3000:
                    output = output[:1500] + "\n...(truncated)...\n" + output[-1500:]
                deps.test_results = output
                logger.info("Test failures detected after file modifications")
        except subprocess.TimeoutExpired:
            deps.test_results = f"Tests timed out after {settings.agent.test_timeout}s"
        except FileNotFoundError:
            pass  # test runner not installed

    async def _clear_modified_files_turn_end(event: TurnEnd) -> None:
        """Clear modified files list at end of turn, after all hooks have read it."""
        deps._files_modified_this_turn.clear()

    async def _clear_modified_files_turn_start(event: TurnStart) -> None:
        deps._files_modified_this_turn.clear()

    hook_registry.on(PostToolUse, _track_modified_files, priority=25)
    hook_registry.on(TurnEnd, _run_tests, priority=25)
    # Clear at priority 50 — after test (25) and critique (30) have both read the list
    hook_registry.on(TurnEnd, _clear_modified_files_turn_end, priority=50)
    hook_registry.on(TurnStart, _clear_modified_files_turn_start, priority=0)


async def _detect_test_command(deps: AgentDeps) -> str | None:
    """Auto-detect the test command for the project. Caches result."""
    if deps.test_command is not None:
        return deps.test_command
    if deps._test_command_searched:
        return None

    deps._test_command_searched = True
    cwd = deps.cwd

    # Check for .forge/test-config.json (same schema as .claude/test-config.json)
    for config_name in (".forge/test-config.json", ".claude/test-config.json"):
        config_path = cwd / config_name
        if config_path.is_file():
            try:
                import json

                data = json.loads(config_path.read_text())
                if data.get("testCommand"):
                    deps.test_command = data["testCommand"]
                    return deps.test_command
            except (json.JSONDecodeError, OSError):
                logger.debug("Failed to read %s", config_name, exc_info=True)

    # Check for common test runners
    if (cwd / "pyproject.toml").exists() or (cwd / "pytest.ini").exists() or (cwd / "setup.cfg").exists():
        # Prefer uv if uv.lock exists (uv-managed project)
        if (cwd / "uv.lock").exists():
            deps.test_command = "uv run pytest -x -q --tb=short --no-header"
        else:
            deps.test_command = "python -m pytest -x -q --tb=short --no-header"
        return deps.test_command
    if (cwd / "package.json").exists():
        deps.test_command = "npm test"
        return deps.test_command
    if (cwd / "Cargo.toml").exists():
        deps.test_command = "cargo test"
        return deps.test_command
    if (cwd / "go.mod").exists():
        deps.test_command = "go test ./..."
        return deps.test_command

    return None


def _scope_test_command(base_cmd: str, modified_files: list[str]) -> str:
    """Scope test command to modified files where possible."""
    if not modified_files:
        return base_cmd

    # For pytest, pass test files or related test files
    if "pytest" in base_cmd:
        test_files = []
        for f in modified_files:
            p = Path(f)
            if p.name.startswith("test_") or p.name.endswith("_test.py"):
                test_files.append(f)
            elif p.suffix == ".py":
                test_name = f"test_{p.name}"
                # Check sibling
                sibling = p.parent / test_name
                if sibling.exists():
                    test_files.append(str(sibling))
                # Walk upward to find project-level tests/ directories
                for ancestor in p.parents:
                    for tests_dir in ("tests", "test", "tests/unit", "tests/integration"):
                        candidate = ancestor / tests_dir / test_name
                        if candidate.exists():
                            test_files.append(str(candidate))
                    # Stop at project root indicators
                    if any((ancestor / marker).exists() for marker in ("pyproject.toml", "setup.py", ".git")):
                        break
        if test_files:
            # Deduplicate while preserving order
            seen: set[str] = set()
            unique = []
            for tf in test_files:
                if tf not in seen:
                    seen.add(tf)
                    unique.append(tf)
            return f"{base_cmd} {' '.join(unique)}"

    return base_cmd


def wire_critique_hooks(hook_registry: HookRegistry, deps: AgentDeps) -> None:
    """Register critique-before-commit hooks.

    At turn end, if 2+ files were modified, generates a diff and sends it to
    the fast model for review. Issues are stored on deps for injection into
    the next turn's system prompt.
    """

    async def _run_critique(event: TurnEnd) -> None:
        if not settings.agent.critique_enabled:
            return
        # Only critique multi-file changes
        real_files = [
            f for f in deps._files_modified_this_turn
            if not f.startswith("__cmd_write:")
        ]
        if len(real_files) < settings.agent.critique_min_writes:
            return

        # Generate diff of changes
        import subprocess

        try:
            result = subprocess.run(
                ["git", "diff", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(deps.cwd),
            )
            diff_text = result.stdout.strip()
            if not diff_text:
                # Try unstaged diff
                result = subprocess.run(
                    ["git", "diff"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(deps.cwd),
                )
                diff_text = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return

        if not diff_text:
            return

        # Truncate diff
        max_chars = settings.agent.critique_max_diff_chars
        if len(diff_text) > max_chars:
            diff_text = diff_text[:max_chars] + "\n... (diff truncated)"

        # Send to fast model for critique
        critique = await _call_critique_model(diff_text, deps)
        if critique and "LGTM" not in critique.upper():
            deps.critique_results = critique
            logger.info("Critique found issues in multi-file changes")

    hook_registry.on(TurnEnd, _run_critique, priority=30)


async def _call_critique_model(diff_text: str, deps: AgentDeps) -> str | None:
    """Send diff to the critique model with CRITIQUE_SYSTEM prompt.

    Uses heavy model by default (configurable via ollama.critique_model).
    If gemini.critique_model is set and Gemini is available, routes to Gemini.
    """
    import httpx

    from forge.prompts.refine import CRITIQUE_SYSTEM

    prompt = f"Review this code diff:\n\n```diff\n{diff_text}\n```"

    # Try Gemini critique if configured (independent of gemini.enabled)
    from forge.agent.gemini import is_gemini_critique_available, mark_rate_limited

    if is_gemini_critique_available():
        try:
            from pydantic_ai import Agent as _Agent

            gemini_model = f"google-gla:{settings.gemini.critique_model}"
            critique_agent = _Agent(model=gemini_model, instructions=CRITIQUE_SYSTEM)
            result = await critique_agent.run(
                prompt, model_settings={"timeout": settings.gemini.timeout},
            )
            logger.info("Gemini critique completed via %s", settings.gemini.critique_model)
            return result.output
        except Exception as exc:
            # Check for rate limiting (429)
            exc_str = str(exc).lower()
            if "429" in exc_str or "rate" in exc_str:
                retry_after = 60.0
                # Try to extract retry-after from error
                import re
                match = re.search(r"retry.?after[:\s]*(\d+)", exc_str)
                if match:
                    retry_after = float(match.group(1))
                mark_rate_limited(retry_after)
            logger.debug("Gemini critique failed, falling back to local", exc_info=True)

    # Local critique: use configured critique_model or heavy_model
    critique_model = settings.ollama.critique_model or settings.ollama.heavy_model
    base_url = settings.ollama.base_url

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{base_url}/api/chat",
                json={
                    "model": critique_model,
                    "messages": [
                        {"role": "system", "content": CRITIQUE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {"num_ctx": 16384},
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("message", {}).get("content", "")
    except Exception:
        logger.debug("Critique model call failed", exc_info=True)

    return None


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
    cloud_info = " | [yellow]cloud reasoning ON[/yellow]" if deps.cloud_reasoning_enabled else ""

    persist_info = f"\nSession: [dim]{session_id[:8]}…[/dim]" if db else ""
    worktree_banner = ""
    if deps.worktree:
        worktree_banner = (
            f"\nWorktree: [cyan]{deps.worktree.branch}[/cyan] "
            f"at [dim]{deps.worktree.path}[/dim]"
        )

    active_model = deps.model_override or settings.ollama.heavy_model
    is_agent = system is AGENT_SYSTEM

    # Build model assignment lines
    model_line = f"Models: [green]{settings.ollama.heavy_model}[/green] (heavy) + [green]{settings.ollama.fast_model}[/green] (fast)"

    # Show role-specific model assignments
    role_parts: list[str] = []
    critique_model = settings.gemini.critique_model
    if critique_model:
        role_parts.append(f"Critique: [cyan]{critique_model}[/cyan]")
    planning_model = settings.gemini.model if settings.gemini.enabled else None
    if planning_model:
        role_parts.append(f"Planning: [cyan]{planning_model}[/cyan]")
    compaction_model = settings.agent.compaction_model or settings.ollama.heavy_model
    role_parts.append(f"Compaction: [green]{compaction_model}[/green]")
    roles_line = " | ".join(role_parts)

    console.print(
        Panel(
            f"[bold]Forge v{__version__}[/bold] — {'agentic coding mode' if is_agent else 'chat + tools'}\n"
            f"{model_line}\n"
            f"{roles_line}\n"
            f"Permissions: [{'green' if deps.permission == PermissionPolicy.YOLO else 'yellow'}]"
            f"{deps.permission.value}[/{'green' if deps.permission == PermissionPolicy.YOLO else 'yellow'}]\n"
            f"{project_info}{instr_info}{rag_info}{memory_info}{mcp_info}{cloud_info}\n"
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

    # Post-session memory summary
    if (
        msg_count >= settings.agent.session_memory_threshold
        and db
        and deps.memory_db
        and deps.memory_project
    ):
        try:
            from forge.agent.memory import get_startup_memories

            rows = await get_startup_memories(deps.memory_db, deps.memory_project, limit=5)
            if rows:
                console.print("[dim]Memories persisted this session:[/dim]")
                for r in rows:
                    console.print(f"  [dim][{r.category}] {r.subject}[/dim]")
        except Exception:
            logger.debug("Session memory summary failed", exc_info=True)

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
