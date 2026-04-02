"""Agent REPL loop — multi-turn agentic coding with tool use."""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from forge import __version__
from forge.agent.context import (
    DEFAULT_TOKEN_BUDGET,
    count_messages_tokens,
    smart_compact_history,
)
from forge.agent.deps import AgentDeps
from forge.agent.hooks import (
    HookRegistry,
    PreToolUse,
    SessionEnd,
    SessionStart,
    TurnEnd,
    TurnStart,
    make_permission_handler,
)
from forge.agent.permissions import PermissionPolicy
from forge.agent.render import render_events
from forge.agent.status import StatusTracker
from forge.agent.task_store import TaskStore
from forge.agent.tools import ALL_TOOLS, MEMORY_TOOLS, TASK_TOOLS
from forge.agent.turn_buffer import TurnBuffer
from forge.config import settings
from forge.log import get_logger
from forge.models.ollama import _ensure_ollama_env

if TYPE_CHECKING:
    from forge.storage.database import Database

logger = get_logger(__name__)

AGENT_SYSTEM = """\
You are Forge, a versatile local AI assistant. You help users with coding, writing, analysis, research, and general questions — as well as reading, understanding, editing, and creating code.

You have access to tools for reading files, writing files, editing files, running shell commands, searching code, listing files, web search, and web fetch. Use them to accomplish the user's request.

## Web research rules
1. Search snippets are often enough — if they answer the question, STOP and respond immediately.
2. Fetch a page ONLY when snippets lack the specific detail needed.
3. Budget per turn: at most 2 web_search + at most 2 web_fetch calls total. If you still lack info after that, answer with what you have and note gaps.
4. Never re-fetch a URL that returned an error.
5. Never fetch more than 3 URLs total in one turn.
6. For well-known facts, history, math, and science fundamentals, answer from your knowledge without searching.

## General guidelines
- Read files before editing them to understand existing code.
- Use edit_file for targeted changes (exact string replacement). Use write_file only for new files or complete rewrites.
- When searching, use search_code with ripgrep patterns. Use list_files to understand project structure.
- Run commands to test changes, run builds, or gather information.
- Be concise in explanations. Show what you did, not what you plan to do.
- When making changes, verify them by reading the modified file or running tests.
- All file paths are relative to the working directory unless absolute.
- IMPORTANT: When you use tools like web_search or web_fetch, the results are raw data for YOU to process. Always synthesize tool results into a direct, natural answer to the user's question. Never dump raw search results or tool output as your response — interpret, summarize, and answer the question.
- If the user asks a question (e.g. "what's the weather?"), use the appropriate tool to gather information, then respond with a clear answer based on what you found.

## Task management
When your work involves 3+ distinct steps, create tasks to track progress.
Mark tasks in_progress when starting, completed when done.
If you discover additional work, create new tasks.
Do not create tasks for single simple operations.

## Memory
When the user asks you to remember something, use save_memory with an appropriate category.
When you need to recall past context, use recall_memories.
Categories: feedback (user corrections), project (decisions/context), user (role/preferences), reference (external pointers).
"""

PLAN_OVERLAY = """\
You are in PLANNING mode. Your task is to analyze the request and produce a structured plan.

Do NOT execute any actions or use any tools. Instead, output a clear plan with:
1. **Goal**: What needs to be accomplished
2. **Steps**: Numbered steps to achieve the goal
3. **Files**: Which files will be read/modified/created
4. **Risks**: Any potential issues or edge cases

Be specific and actionable. The plan will be reviewed before execution.
"""


def create_agent(
    system: str = AGENT_SYSTEM,
    cwd: Path | None = None,
    model: str | None = None,
    tools: list[Tool] | None = None,
    toolsets: list | None = None,
) -> Agent[AgentDeps, str]:
    """Create a pydantic-ai Agent with coding tools."""
    from forge.core.project import build_project_context

    _ensure_ollama_env()

    full_system = system
    if cwd:
        full_system += "\n\n" + build_project_context(cwd)

    model_name = model or settings.ollama.heavy_model

    return Agent(
        model=f"ollama:{model_name}",
        instructions=full_system,
        tools=tools if tools is not None else ALL_TOOLS,
        toolsets=toolsets or [],
        deps_type=AgentDeps,
        model_settings=ModelSettings(timeout=300),
        retries=3,
    )


async def _run_with_status(
    agent: Agent[AgentDeps, str],
    prompt: str,
    deps: AgentDeps,
    message_history: list[ModelMessage] | None,
    turn_number: int = 0,
) -> list[ModelMessage]:
    """Run agent with status tracker lifecycle management.

    Returns the updated message history.
    """
    tracker = StatusTracker(console=deps.console, visible=deps.status_visible)
    deps.status_tracker = tracker

    # Create turn buffer for this turn
    turn_buffer = TurnBuffer(console=deps.console)
    deps.turn_buffer = turn_buffer

    # Show active task in status detail
    if deps.task_store:
        active_task = deps.task_store.get_active()
        if active_task and active_task.active_form:
            tracker.set_phase(tracker._phase, active_task.active_form)

    def _on_toggle(visible: bool) -> None:
        deps.status_visible = visible

    def _on_tools_toggle() -> None:
        deps.tools_visible = not deps.tools_visible

    tracker.start(on_toggle=_on_toggle, on_tools_toggle=_on_tools_toggle)

    # Emit TurnStart
    registry = deps.hook_registry
    if registry:
        await registry.emit(TurnStart(turn_number=turn_number, prompt=prompt))

    import time as _time
    turn_start = _time.monotonic()

    try:
        run_kwargs: dict = dict(
            deps=deps,
            message_history=message_history,
            event_stream_handler=render_events,
            usage_limits=UsageLimits(request_limit=15),
        )
        if deps.model_override:
            run_kwargs["model"] = f"ollama:{deps.model_override}"

        result = await agent.run(prompt, **run_kwargs)
        # Pause tracker before printing to prevent status line garbling output
        tracker.pause()

        # Add final answer + summary to turn buffer, then print via buffer
        if isinstance(result.output, str) and result.output.strip():
            turn_buffer.add(Markdown(result.output), is_tool=False)
        tracker.stop()
        turn_buffer.add(tracker.summary(), is_tool=False)
        turn_buffer.print_final(deps.tools_visible)

        # Emit TurnEnd
        if registry:
            elapsed = _time.monotonic() - turn_start
            await registry.emit(TurnEnd(
                turn_number=turn_number,
                tool_call_count=tracker.tool_calls,
                elapsed=elapsed,
            ))

        deps.status_tracker = None
        return result.all_messages()
    except BaseException:
        tracker.stop()
        deps.console.print(tracker.summary())
        deps.status_tracker = None
        deps.turn_buffer = None
        raise


async def _plan_and_execute(
    agent: Agent[AgentDeps, str],
    prompt: str,
    deps: AgentDeps,
    message_history: list[ModelMessage] | None,
) -> list[ModelMessage] | None:
    """Two-phase plan-then-execute workflow.

    Phase 1: Generate a plan (no tools).
    Phase 2: On approval, execute with full tools.
    Returns updated message_history, or the original if cancelled.
    """
    console = deps.console

    # Phase 1: Planning — create a no-tools agent with plan overlay
    _ensure_ollama_env()
    plan_agent: Agent[AgentDeps, str] = Agent(
        model=f"ollama:{settings.ollama.heavy_model}",
        instructions=PLAN_OVERLAY,
        tools=[],  # No tools in planning mode
        deps_type=AgentDeps,
        model_settings=ModelSettings(timeout=300),
    )

    console.print("[dim]Planning...[/dim]")
    try:
        plan_result = await plan_agent.run(
            prompt,
            deps=deps,
            message_history=None,  # Fresh context for planning
        )
        plan_text = plan_result.output
    except Exception as e:
        _handle_agent_error(console, e)
        return message_history

    # Display the plan
    console.print(
        Panel(
            plan_text,
            title="[bold]Plan[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # Phase 2: Ask for approval
    try:
        from prompt_toolkit import prompt as pt_prompt

        answer = await asyncio.get_running_loop().run_in_executor(
            None, lambda: pt_prompt("Execute this plan? [Y/n] ")
        )
        answer = answer.strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("[dim]Plan cancelled.[/dim]")
        return message_history

    if answer in ("n", "no"):
        console.print("[dim]Plan cancelled.[/dim]")
        return message_history

    # Phase 3: Execute with full agent
    deps.active_plan = plan_text
    execution_prompt = (
        f"Execute this plan step by step:\n\n{plan_text}\n\n"
        f"Original request: {prompt}\n\n"
        "As you complete each step, note it as done. If a step fails, "
        "explain why and adjust your approach before continuing."
    )
    try:
        result = await _run_with_status(agent, execution_prompt, deps, message_history)
    finally:
        deps.active_plan = None
    return result


from pydantic import TypeAdapter

_message_list_adapter = TypeAdapter(list[ModelMessage])


async def _save_agent_session(
    db: Database,
    session_id: str,
    messages: list[ModelMessage],
) -> None:
    """Persist agent message history as JSON to the conversations table."""
    try:
        history_json = _message_list_adapter.dump_json(messages).decode()
        await db.delete_agent_history(session_id)
        await db.save_message(session_id, "agent_history", history_json, model="")
    except Exception:
        logger.debug("Failed to save session", exc_info=True)


async def _load_agent_history(
    db: Database,
    session_id: str,
) -> list[ModelMessage] | None:
    """Load agent message history from the database. Returns None if not found."""
    try:
        raw = await db.load_agent_history(session_id)
        if raw is None:
            return None
        return _message_list_adapter.validate_json(raw)
    except Exception:
        logger.debug("Failed to load history", exc_info=True)
        return None


def _rebuild_agent(
    deps: AgentDeps,
    system: str,
    extra_tools: list[Tool] | None = None,
    toolsets: list | None = None,
) -> Agent[AgentDeps, str]:
    """Rebuild agent with fresh project context for current deps.cwd."""
    tools = ALL_TOOLS + (extra_tools or [])
    return create_agent(system=system, cwd=deps.cwd, tools=tools, toolsets=toolsets)


async def agent_repl(
    initial_prompt: str | None = None,
    permission: PermissionPolicy | None = None,
    resume_session_id: str | None = None,
    system: str = AGENT_SYSTEM,
    worktree_name: str | None = None,
) -> None:
    """Run the agentic REPL with tool use."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings

    console = Console()
    cwd = Path.cwd()

    # Create worktree if requested via CLI flag
    worktree_info = None
    if worktree_name is not None:
        from forge.agent.worktree import create_worktree, is_git_repo

        if not is_git_repo(cwd):
            console.print("[red]Not a git repository — cannot create worktree.[/red]")
            return
        try:
            worktree_info = create_worktree(cwd, worktree_name or None)
            cwd = worktree_info.path
            console.print(
                f"[green]Worktree created:[/green] {worktree_info.path}\n"
                f"[green]Branch:[/green] {worktree_info.branch}"
            )
        except RuntimeError as e:
            console.print(f"[red]Worktree error:[/red] {e}")
            return

    deps = AgentDeps(
        cwd=cwd,
        console=console,
        permission=permission or PermissionPolicy.AUTO,
        worktree=worktree_info,
    )

    # Hook registry — permission enforcement
    hook_registry = HookRegistry()
    hook_registry.on(PreToolUse, make_permission_handler(deps))
    deps.hook_registry = hook_registry

    # Task tracking — always available
    deps.task_store = TaskStore()

    agent = create_agent(system=system, cwd=cwd)
    message_history: list[ModelMessage] | None = None

    # Custom key bindings
    kb = KeyBindings()

    @kb.add("c-o")
    def _toggle_status(event):
        deps.status_visible = not deps.status_visible
        state = "visible" if deps.status_visible else "hidden"
        # Update any active status tracker
        if deps.status_tracker:
            deps.status_tracker.visible = deps.status_visible
            if not deps.status_visible:
                deps.status_tracker._clear_line()
        sys.stderr.write(f"\r\033[2K\033[2mStatus line: {state} (Ctrl-O to toggle)\033[0m\n")
        sys.stderr.flush()

    @kb.add("c-r")
    def _toggle_tools(event):
        deps.tools_visible = not deps.tools_visible
        if deps.turn_buffer and deps.turn_buffer._items:
            # Erase prompt_toolkit's prompt display before rerendering
            event.app.renderer.erase()
            deps.turn_buffer.rerender(deps.tools_visible)
            # Force prompt_toolkit to redraw the prompt
            event.app.renderer.reset()
            event.app.invalidate()
        else:
            state = "visible" if deps.tools_visible else "hidden"
            sys.stderr.write(f"\r\033[2K\033[2mTool results: {state} (Ctrl-R to toggle)\033[0m\n")
            sys.stderr.flush()

    session: PromptSession[str] = PromptSession(
        history=InMemoryHistory(), key_bindings=kb
    )

    # DB persistence
    db = await _connect_db()
    session_id = resume_session_id or str(uuid.uuid4())

    if db and resume_session_id:
        # Resume: load existing message history
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
        try:
            await db.create_session(session_id, mode="agent" if system is AGENT_SYSTEM else "chat")
        except Exception:
            logger.warning("Session creation failed, disabling persistence", exc_info=True)
            db = None

    # Memory integration — wire DB for cross-session memory
    memory_count = 0
    if db:
        deps.memory_db = db
        deps.memory_project = cwd.name
        try:
            memory_count = await db.count_memories(cwd.name)
        except Exception:
            logger.debug("Memory count failed", exc_info=True)

    # RAG integration — check if cwd is indexed
    # When in a worktree, use the base repo name for project lookup
    rag_project_name = worktree_info.base_dir.name if worktree_info else cwd.name
    rag_available = False
    if db:
        try:
            stats = await db.get_project_stats(rag_project_name)
            if stats.get("chunk_count", 0) > 0:
                deps.rag_db = db
                deps.rag_project = rag_project_name
                rag_available = True
        except Exception:
            logger.debug("RAG stats check failed", exc_info=True)

    # MCP server integration
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
    mcp_servers = mcp_started  # Only pass successfully started servers

    # Build agent with conditional tools
    extra_tools: list[Tool] = list(TASK_TOOLS)  # Always add task tools
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
    agent = create_agent(
        system=system, cwd=cwd, tools=ALL_TOOLS + extra_tools, toolsets=mcp_servers or None,
    )

    # Inject dynamic task/memory context into system prompt
    @agent.system_prompt
    async def _inject_tasks(ctx: RunContext[AgentDeps]) -> str:
        if ctx.deps.task_store:
            return ctx.deps.task_store.to_prompt()
        return ""

    @agent.system_prompt
    async def _inject_memories(ctx: RunContext[AgentDeps]) -> str:
        if ctx.deps.memory_db and ctx.deps.memory_project:
            try:
                from forge.agent.memory import get_startup_memories
                rows = await get_startup_memories(ctx.deps.memory_db, ctx.deps.memory_project, limit=10)
                if rows:
                    lines = ["## Remembered context\n"]
                    for r in rows:
                        lines.append(f"- [{r.category}] **{r.subject}**: {r.content}")
                    return "\n".join(lines)
            except Exception:
                logger.debug("Memory injection failed", exc_info=True)
        return ""

    # Emit SessionStart
    await hook_registry.emit(SessionStart(
        session_id=session_id,
        cwd=str(cwd),
        permission=deps.permission.value,
    ))

    # Show project context info
    from forge.core.project import INSTRUCTION_FILES, detect_project_type

    project_type = detect_project_type(cwd)
    project_info = f"Project: [cyan]{project_type}[/cyan] | " if project_type else ""
    instructions_loaded = any((cwd / f).is_file() for f in INSTRUCTION_FILES)
    instr_info = "[green]instructions loaded[/green]" if instructions_loaded else ""
    rag_info = " | [green]RAG indexed[/green]" if rag_available else ""
    memory_info = f" | [green]{memory_count} memories[/green]" if memory_count else ""
    mcp_info = f" | [green]{len(mcp_servers)} MCP server(s)[/green]" if mcp_servers else ""

    persist_info = f"\nSession: [dim]{session_id[:8]}…[/dim]" if db else ""
    worktree_banner = ""
    if deps.worktree:
        worktree_banner = f"\nWorktree: [cyan]{deps.worktree.branch}[/cyan] at [dim]{deps.worktree.path}[/dim]"

    active_model = deps.model_override or settings.ollama.heavy_model
    console.print(
        Panel(
            f"[bold]Forge v{__version__}[/bold] — {'agentic coding mode' if system is AGENT_SYSTEM else 'chat + tools'}\n"
            f"Model: [green]{active_model}[/green]\n"
            f"Permissions: [{'green' if deps.permission == PermissionPolicy.YOLO else 'yellow'}]"
            f"{deps.permission.value}[/{'green' if deps.permission == PermissionPolicy.YOLO else 'yellow'}]\n"
            f"{project_info}{instr_info}{rag_info}{memory_info}{mcp_info}\n"
            f"Working directory: [dim]{deps.cwd}[/dim]"
            f"{worktree_banner}"
            f"{persist_info}\n"
            "Type [bold]/help[/bold] for commands, [bold]Ctrl-O[/bold] status, [bold]Ctrl-R[/bold] tools, [bold]Ctrl-D[/bold] exit",
            title="forge agent" if system is AGENT_SYSTEM else "forge",
            border_style="magenta",
        )
    )

    # Handle initial prompt if provided
    turn_counter = 0
    if initial_prompt:
        turn_counter += 1
        console.print(f"\n[bold]> {initial_prompt}[/bold]")
        try:
            prompt = _maybe_prepend_think(initial_prompt, deps)
            message_history = await _run_with_status(agent, prompt, deps, message_history, turn_number=turn_counter)
            # Incremental save
            if db and message_history:
                asyncio.create_task(_save_agent_session(db, session_id, message_history))
            if db:
                # Auto-title from first prompt
                title = initial_prompt[:60].strip()
                if len(initial_prompt) > 60:
                    title = title.rsplit(" ", 1)[0] + "…"
                try:
                    await db.update_session_title(session_id, title)
                except Exception:
                    logger.debug("Title update failed", exc_info=True)
        except Exception as e:
            _handle_agent_error(console, e)

    # REPL loop
    title_set = initial_prompt is not None
    try:
        while True:
            try:
                user_input = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: session.prompt("\n❯ ")
                )
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/dim]")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # REPL commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                if cmd in ("/quit", "/exit", "/q"):
                    console.print("[dim]Goodbye.[/dim]")
                    break
                elif cmd == "/clear":
                    message_history = None
                    console.print("[dim]Conversation cleared.[/dim]")
                    continue
                elif cmd == "/compact":
                    if message_history:
                        before = len(message_history)
                        _, before_tokens = count_messages_tokens(message_history)
                        console.print(
                            f"[dim]Compacting {before} messages (~{before_tokens:,} tokens)...[/dim]"
                        )
                        message_history = await smart_compact_history(message_history)
                        after = len(message_history)
                        _, after_tokens = count_messages_tokens(message_history)
                        console.print(
                            f"[dim]Compacted: {before} → {after} messages "
                            f"(~{before_tokens:,} → ~{after_tokens:,} tokens)[/dim]"
                        )
                    else:
                        console.print("[dim]No history to compact.[/dim]")
                    continue
                elif cmd == "/tokens":
                    if message_history:
                        count, tokens = count_messages_tokens(message_history)
                        console.print(f"[dim]{count} messages, ~{tokens:,} tokens[/dim]")
                    else:
                        console.print("[dim]No history yet.[/dim]")
                    continue
                elif cmd == "/messages":
                    if message_history:
                        for i, msg in enumerate(message_history):
                            kind = type(msg).__name__
                            text_len = len(str(msg))
                            console.print(f"[dim]  {i}: {kind} ({text_len} chars)[/dim]")
                    else:
                        console.print("[dim]No history yet.[/dim]")
                    continue
                elif cmd == "/status":
                    deps.status_visible = not deps.status_visible
                    if deps.status_tracker:
                        deps.status_tracker.visible = deps.status_visible
                        if not deps.status_visible:
                            deps.status_tracker._clear_line()
                    state = "visible" if deps.status_visible else "hidden"
                    console.print(f"[dim]Status line: {state}[/dim]")
                    continue
                elif cmd == "/think":
                    deps.thinking_enabled = not deps.thinking_enabled
                    state = "on" if deps.thinking_enabled else "off"
                    console.print(f"[dim]Extended thinking: {state}[/dim]")
                    continue
                elif cmd == "/plan":
                    # Extract prompt after /plan
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[dim]Usage: /plan <prompt>[/dim]")
                        continue
                    plan_prompt = parts[1]
                    try:
                        result = await _plan_and_execute(
                            agent, plan_prompt, deps, message_history
                        )
                        if result is not None:
                            message_history = result
                    except KeyboardInterrupt:
                        console.print("\n[dim]Interrupted.[/dim]")
                    except Exception as e:
                        _handle_agent_error(console, e)
                    continue
                elif cmd == "/tools":
                    deps.tools_visible = not deps.tools_visible
                    if deps.turn_buffer and deps.turn_buffer._items:
                        deps.turn_buffer.rerender(deps.tools_visible)
                    else:
                        state = "visible" if deps.tools_visible else "hidden"
                        console.print(f"[dim]Tool results: {state}[/dim]")
                    continue
                elif cmd == "/model":
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        current = deps.model_override or settings.ollama.heavy_model
                        console.print(
                            f"[dim]Current model: {current}\n"
                            f"Usage: /model fast | /model heavy | /model <name>[/dim]"
                        )
                        continue
                    arg = parts[1].strip()
                    if arg == "heavy":
                        deps.model_override = None
                        console.print(f"[dim]Model: {settings.ollama.heavy_model} (heavy)[/dim]")
                    elif arg == "fast":
                        deps.model_override = settings.ollama.fast_model
                        console.print(f"[dim]Model: {settings.ollama.fast_model} (fast)[/dim]")
                    else:
                        deps.model_override = arg
                        console.print(f"[dim]Model: {arg}[/dim]")
                    continue
                elif cmd == "/plan-status":
                    if deps.active_plan:
                        console.print(
                            Panel(deps.active_plan, title="Active Plan", border_style="cyan")
                        )
                    else:
                        console.print("[dim]No active plan.[/dim]")
                    continue
                elif cmd == "/tasks":
                    if deps.task_store:
                        tasks = deps.task_store.list_all()
                        if tasks:
                            from rich.table import Table
                            table = Table(title="Tasks", border_style="dim")
                            table.add_column("ID", style="cyan")
                            table.add_column("Status")
                            table.add_column("Subject")
                            table.add_column("Blocked By", style="dim")
                            for t in tasks:
                                status_style = {
                                    "pending": "dim",
                                    "in_progress": "yellow",
                                    "completed": "green",
                                }.get(t.status.value, "dim")
                                blocked = ", ".join(t.blocked_by) if t.blocked_by else ""
                                table.add_row(t.id, f"[{status_style}]{t.status.value}[/{status_style}]", t.subject, blocked)
                            console.print(table)
                        else:
                            console.print("[dim]No tasks.[/dim]")
                    else:
                        console.print("[dim]Task tracking unavailable.[/dim]")
                    continue
                elif cmd == "/memory":
                    parts = user_input.split(maxsplit=2)
                    if len(parts) >= 3 and parts[1].lower() == "search":
                        query = parts[2]
                        if deps.memory_db and deps.memory_project:
                            try:
                                from forge.agent.memory import recall_from_db
                                rows = await recall_from_db(deps.memory_db, deps.memory_project, query)
                                if rows:
                                    for r in rows:
                                        score = f"{r.score:.2f}" if r.score else ""
                                        console.print(f"  [cyan]#{r.id}[/cyan] [{r.category}] [bold]{r.subject}[/bold]: {r.content} [dim]{score}[/dim]")
                                else:
                                    console.print("[dim]No matching memories.[/dim]")
                            except Exception as e:
                                console.print(f"[red]Memory search error:[/red] {e}")
                        else:
                            console.print("[dim]Memory unavailable (no database).[/dim]")
                    else:
                        # Show memory stats
                        if deps.memory_db and deps.memory_project:
                            try:
                                count = await deps.memory_db.count_memories(deps.memory_project)
                                console.print(f"[dim]{count} memories for project '{deps.memory_project}'[/dim]")
                                console.print("[dim]Usage: /memory search <query>[/dim]")
                            except Exception:
                                console.print("[dim]Memory stats unavailable.[/dim]")
                        else:
                            console.print("[dim]Memory unavailable (no database).[/dim]")
                    continue
                elif cmd == "/forget":
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[dim]Usage: /forget <memory_id>[/dim]")
                        continue
                    try:
                        mid = int(parts[1].strip())
                        if deps.memory_db:
                            deleted = await deps.memory_db.delete_memory(mid)
                            if deleted:
                                console.print(f"[dim]Deleted memory #{mid}.[/dim]")
                            else:
                                console.print(f"[dim]Memory #{mid} not found.[/dim]")
                        else:
                            console.print("[dim]Memory unavailable (no database).[/dim]")
                    except ValueError:
                        console.print("[dim]Usage: /forget <memory_id> (numeric ID)[/dim]")
                    continue
                elif cmd == "/mcp":
                    if mcp_servers:
                        for server in mcp_servers:
                            sid = getattr(server, "id", None) or "unknown"
                            stype = type(server).__name__
                            is_up = getattr(server, "is_running", False)
                            state = "running" if is_up else "stopped"
                            console.print(
                                f"  [cyan]{sid}[/cyan] — {stype} [{state}]"
                            )
                    else:
                        console.print("[dim]No MCP servers configured.[/dim]")
                    continue
                elif cmd == "/help":
                    console.print(
                        Panel(
                            "/clear       — clear conversation history\n"
                            "/compact     — compact history to fit token budget\n"
                            "/tokens      — show estimated token count\n"
                            "/messages    — list message history\n"
                            "/model       — switch model (fast/heavy/<name>)\n"
                            "/status      — toggle status line (or Ctrl-O)\n"
                            "/tools       — toggle tool result display (or Ctrl-R)\n"
                            "/think       — toggle extended thinking on/off\n"
                            "/plan        — plan before executing (e.g. /plan refactor X)\n"
                            "/plan-status — show active plan\n"
                            "/tasks       — show task list\n"
                            "/mcp         — list connected MCP servers\n"
                            "/memory      — show memory stats / search memories\n"
                            "/forget <id> — delete a memory by ID\n"
                            "/cwd         — show working directory\n"
                            "/cd <dir>    — change working directory\n"
                            "/worktree    — create isolated git worktree\n"
                            "/quit        — exit",
                            title="commands",
                            border_style="dim",
                        )
                    )
                    continue
                elif cmd == "/cwd":
                    console.print(f"[dim]{deps.cwd}[/dim]")
                    continue
                elif cmd == "/cd":
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[dim]Usage: /cd <directory>[/dim]")
                        continue
                    new_dir = Path(parts[1]).expanduser()
                    if not new_dir.is_absolute():
                        new_dir = deps.cwd / new_dir
                    new_dir = new_dir.resolve()
                    if new_dir.is_dir():
                        deps.cwd = new_dir
                        agent = _rebuild_agent(
                            deps, system, extra_tools,
                            toolsets=mcp_servers or None,
                        )
                        console.print(f"[dim]Working directory: {deps.cwd} (agent reloaded)[/dim]")
                    else:
                        console.print(f"[red]Not a directory: {new_dir}[/red]")
                    continue
                elif cmd == "/worktree":
                    if deps.worktree:
                        console.print("[yellow]Already in a worktree.[/yellow]")
                        continue
                    from forge.agent.worktree import create_worktree, is_git_repo

                    if not is_git_repo(deps.cwd):
                        console.print("[red]Not a git repository — cannot create worktree.[/red]")
                        continue
                    wt_parts = user_input.split(maxsplit=1)
                    wt_name = wt_parts[1].strip() if len(wt_parts) > 1 else None
                    try:
                        wt_info = create_worktree(deps.cwd, wt_name)
                        deps.worktree = wt_info
                        deps.cwd = wt_info.path
                        agent = _rebuild_agent(
                            deps, system, extra_tools,
                            toolsets=mcp_servers or None,
                        )
                        console.print(
                            f"[green]Worktree created:[/green] {wt_info.path}\n"
                            f"[green]Branch:[/green] {wt_info.branch}"
                        )
                    except RuntimeError as e:
                        console.print(f"[red]Worktree error:[/red] {e}")
                    continue
                else:
                    console.print(f"[dim]Unknown command: {cmd}[/dim]")
                    continue

            # Auto-compact if history is getting large
            if message_history and len(message_history) > 40:
                _, tokens = count_messages_tokens(message_history)
                if tokens > DEFAULT_TOKEN_BUDGET * 0.8:
                    before = len(message_history)
                    message_history = await smart_compact_history(message_history)
                    after = len(message_history)
                    console.print(
                        f"[dim]Auto-compacted: {before} → {after} messages[/dim]"
                    )

            # Auto-title from first user message
            if db and not title_set:
                title = user_input[:60].strip()
                if len(user_input) > 60:
                    title = title.rsplit(" ", 1)[0] + "…"
                try:
                    await db.update_session_title(session_id, title)
                except Exception:
                    logger.debug("Title update failed", exc_info=True)
                title_set = True

            # Run agent with status tracking
            try:
                turn_counter += 1
                prompt = _maybe_prepend_think(user_input, deps)
                message_history = await _run_with_status(
                    agent, prompt, deps, message_history, turn_number=turn_counter,
                )
                # Incremental save after each successful turn
                if db and message_history:
                    asyncio.create_task(_save_agent_session(db, session_id, message_history))
                # Persist task store
                if db and deps.task_store:
                    try:
                        await db.save_task_store(session_id, deps.task_store.to_json())
                    except Exception:
                        logger.debug("Task store save failed", exc_info=True)
            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted.[/dim]")
            except Exception as e:
                _handle_agent_error(console, e)
    finally:
        # Emit SessionEnd
        msg_count = len(message_history) if message_history else 0
        await hook_registry.emit(SessionEnd(session_id=session_id, message_count=msg_count))

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


def _maybe_prepend_think(prompt: str, deps: AgentDeps) -> str:
    """Prepend /think or /no_think tag based on thinking_enabled setting."""
    if deps.thinking_enabled:
        return f"/think\n{prompt}"
    return prompt


async def _connect_db():
    """Connect to DB for persistence. Returns None on failure."""
    if not settings.persist_history:
        return None
    try:
        from forge.storage.database import Database

        db = Database()
        await db.connect()
        return db
    except Exception:
        logger.info("Database unavailable — persistence disabled", exc_info=True)
        return None


def _handle_agent_error(console: Console, e: Exception) -> None:
    """Print a user-friendly agent error message."""
    from pydantic_ai.exceptions import UsageLimitExceeded

    if isinstance(e, UsageLimitExceeded):
        console.print(
            "[yellow]Agent hit the request limit (15 iterations).[/yellow] "
            "This usually means the model was stuck in a loop. "
            "The partial result has been preserved in history."
        )
        return

    err_str = str(e).lower()
    if "connection" in err_str or "connect" in err_str:
        console.print(
            "[red]Cannot connect to Ollama.[/red] Is it running? "
            "Check with: [bold]systemctl status ollama[/bold]"
        )
    elif "timeout" in err_str or "timed out" in err_str:
        console.print(
            "[red]Request timed out.[/red] "
            "The model may still be loading — try again in a moment."
        )
    elif "404" in err_str:
        console.print(
            "[red]Model not found.[/red] "
            f"Pull it with: [bold]ollama pull {settings.ollama.heavy_model}[/bold]"
        )
    else:
        console.print(f"[red]Agent error:[/red] {e}")
