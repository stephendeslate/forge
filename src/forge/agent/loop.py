"""Agent REPL loop — multi-turn agentic coding with tool use."""

from __future__ import annotations

import asyncio

import os
import sys
import uuid
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from forge import __version__
from forge.agent.context import count_messages_tokens, smart_compact_history
from forge.agent.deps import AgentDeps
from forge.agent.permissions import PermissionPolicy
from forge.agent.render import render_events
from forge.agent.status import StatusTracker
from forge.agent.tools import ALL_TOOLS
from forge.agent.turn_buffer import TurnBuffer
from forge.config import settings

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


def _ensure_ollama_env() -> None:
    """Set OLLAMA_BASE_URL env var for pydantic-ai's Ollama provider."""
    if "OLLAMA_BASE_URL" not in os.environ:
        base = settings.ollama.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        os.environ["OLLAMA_BASE_URL"] = base


def create_agent(
    system: str = AGENT_SYSTEM,
    cwd: Path | None = None,
    model: str | None = None,
    tools: list | None = None,
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
        deps_type=AgentDeps,
        model_settings=ModelSettings(timeout=300),
        retries=3,
    )


async def _run_with_status(
    agent: Agent[AgentDeps, str],
    prompt: str,
    deps: AgentDeps,
    message_history: list[ModelMessage] | None,
) -> list[ModelMessage]:
    """Run agent with status tracker lifecycle management.

    Returns the updated message history.
    """
    tracker = StatusTracker(console=deps.console, visible=deps.status_visible)
    deps.status_tracker = tracker

    # Create turn buffer for this turn
    turn_buffer = TurnBuffer(console=deps.console)
    deps.turn_buffer = turn_buffer

    def _on_toggle(visible: bool) -> None:
        deps.status_visible = visible

    def _on_tools_toggle() -> None:
        deps.tools_visible = not deps.tools_visible

    tracker.start(on_toggle=_on_toggle, on_tools_toggle=_on_tools_toggle)

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

        answer = await asyncio.get_event_loop().run_in_executor(
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
    db: object,
    session_id: str,
    messages: list[ModelMessage],
) -> None:
    """Persist agent message history as JSON to the conversations table."""
    try:
        history_json = _message_list_adapter.dump_json(messages).decode()
        await db.delete_agent_history(session_id)  # type: ignore[union-attr]
        await db.save_message(session_id, "agent_history", history_json, model="")  # type: ignore[union-attr]
    except Exception:
        pass  # Best-effort persistence


async def _load_agent_history(
    db: object,
    session_id: str,
) -> list[ModelMessage] | None:
    """Load agent message history from the database. Returns None if not found."""
    try:
        raw = await db.load_agent_history(session_id)  # type: ignore[union-attr]
        if raw is None:
            return None
        return _message_list_adapter.validate_json(raw)
    except Exception:
        return None


async def agent_repl(
    initial_prompt: str | None = None,
    permission: PermissionPolicy | None = None,
    resume_session_id: str | None = None,
    system: str = AGENT_SYSTEM,
) -> None:
    """Run the agentic REPL with tool use."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings

    console = Console()
    cwd = Path.cwd()
    agent = create_agent(system=system, cwd=cwd)
    deps = AgentDeps(
        cwd=cwd,
        console=console,
        permission=permission or PermissionPolicy.AUTO,
    )
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
    elif db:
        try:
            await db.create_session(session_id, mode="agent" if system is AGENT_SYSTEM else "chat")
        except Exception:
            db = None

    # RAG integration — check if cwd is indexed
    rag_available = False
    if db:
        try:
            stats = await db.get_project_stats(cwd.name)
            if stats.get("chunk_count", 0) > 0:
                deps.rag_db = db
                deps.rag_project = cwd.name
                rag_available = True
        except Exception:
            pass

    # Build agent with RAG tool if available
    if rag_available:
        from forge.agent.tools import rag_search

        agent = create_agent(
            system=system + "\n\nUse rag_search for semantic code search (conceptual queries). "
            "Use search_code for exact text/regex matches.",
            cwd=cwd,
            tools=ALL_TOOLS + [rag_search],
        )

    # Show project context info
    from forge.core.project import INSTRUCTION_FILES, detect_project_type

    project_type = detect_project_type(cwd)
    project_info = f"Project: [cyan]{project_type}[/cyan] | " if project_type else ""
    instructions_loaded = any((cwd / f).is_file() for f in INSTRUCTION_FILES)
    instr_info = "[green]instructions loaded[/green]" if instructions_loaded else ""
    rag_info = " | [green]RAG indexed[/green]" if rag_available else ""

    persist_info = f"\nSession: [dim]{session_id[:8]}…[/dim]" if db else ""

    active_model = deps.model_override or settings.ollama.heavy_model
    console.print(
        Panel(
            f"[bold]Forge v{__version__}[/bold] — {'agentic coding mode' if system is AGENT_SYSTEM else 'chat + tools'}\n"
            f"Model: [green]{active_model}[/green]\n"
            f"Permissions: [{'green' if deps.permission == PermissionPolicy.YOLO else 'yellow'}]"
            f"{deps.permission.value}[/{'green' if deps.permission == PermissionPolicy.YOLO else 'yellow'}]\n"
            f"{project_info}{instr_info}{rag_info}\n"
            f"Working directory: [dim]{deps.cwd}[/dim]"
            f"{persist_info}\n"
            "Type [bold]/help[/bold] for commands, [bold]Ctrl-O[/bold] status, [bold]Ctrl-R[/bold] tools, [bold]Ctrl-D[/bold] exit",
            title="forge agent" if system is AGENT_SYSTEM else "forge",
            border_style="magenta",
        )
    )

    # Handle initial prompt if provided
    if initial_prompt:
        console.print(f"\n[bold]> {initial_prompt}[/bold]")
        try:
            prompt = _maybe_prepend_think(initial_prompt, deps)
            message_history = await _run_with_status(agent, prompt, deps, message_history)
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
                    pass
        except Exception as e:
            _handle_agent_error(console, e)

    # REPL loop
    title_set = initial_prompt is not None
    try:
        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
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
                        console.print("[dim]Compacting with LLM summarization...[/dim]")
                        message_history = await smart_compact_history(message_history)
                        after = len(message_history)
                        _, tokens = count_messages_tokens(message_history)
                        console.print(
                            f"[dim]Compacted: {before} → {after} messages (~{tokens:,} tokens)[/dim]"
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
                elif cmd == "/help":
                    console.print(
                        Panel(
                            "/clear    — clear conversation history\n"
                            "/compact  — compact history to fit token budget\n"
                            "/tokens   — show estimated token count\n"
                            "/messages — list message history\n"
                            "/model    — switch model (fast/heavy/<name>)\n"
                            "/status   — toggle status line (or Ctrl-O)\n"
                            "/tools    — toggle tool result display (or Ctrl-R)\n"
                            "/think    — toggle extended thinking on/off\n"
                            "/plan     — plan before executing (e.g. /plan refactor X)\n"
                            "/plan-status — show active plan\n"
                            "/cwd      — show working directory\n"
                            "/cd <dir> — change working directory\n"
                            "/quit     — exit",
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
                        console.print(f"[dim]Working directory: {deps.cwd}[/dim]")
                    else:
                        console.print(f"[red]Not a directory: {new_dir}[/red]")
                    continue
                else:
                    console.print(f"[dim]Unknown command: {cmd}[/dim]")
                    continue

            # Auto-compact if history is getting large
            if message_history and len(message_history) > 40:
                message_history = await smart_compact_history(message_history)

            # Auto-title from first user message
            if db and not title_set:
                title = user_input[:60].strip()
                if len(user_input) > 60:
                    title = title.rsplit(" ", 1)[0] + "…"
                try:
                    await db.update_session_title(session_id, title)
                except Exception:
                    pass
                title_set = True

            # Run agent with status tracking
            try:
                prompt = _maybe_prepend_think(user_input, deps)
                message_history = await _run_with_status(
                    agent, prompt, deps, message_history
                )
                # Incremental save after each successful turn
                if db and message_history:
                    asyncio.create_task(_save_agent_session(db, session_id, message_history))
            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted.[/dim]")
            except Exception as e:
                _handle_agent_error(console, e)
    finally:
        # Persist final message history
        if db and message_history:
            await _save_agent_session(db, session_id, message_history)
        if db:
            await db.close()


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
