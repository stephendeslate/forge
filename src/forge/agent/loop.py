"""Agent REPL loop — multi-turn agentic coding with tool use."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings
from rich.console import Console
from rich.panel import Panel

from forge import __version__
from forge.agent.context import compact_history, count_messages_tokens
from forge.agent.deps import AgentDeps
from forge.agent.permissions import PermissionPolicy
from forge.agent.render import render_events
from forge.agent.tools import ALL_TOOLS
from forge.config import settings

AGENT_SYSTEM = """\
You are Forge, a versatile local AI assistant. You help users with coding, writing, analysis, research, and general questions — as well as reading, understanding, editing, and creating code.

You have access to tools for reading files, writing files, editing files, running shell commands, searching code, and listing files. Use them to accomplish the user's request.

Guidelines:
- Read files before editing them to understand existing code.
- Use edit_file for targeted changes (exact string replacement). Use write_file only for new files or complete rewrites.
- When searching, use search_code with ripgrep patterns. Use list_files to understand project structure.
- Run commands to test changes, run builds, or gather information.
- Be concise in explanations. Show what you did, not what you plan to do.
- When making changes, verify them by reading the modified file or running tests.
- All file paths are relative to the working directory unless absolute.
"""

def _ensure_ollama_env() -> None:
    """Set OLLAMA_BASE_URL env var for pydantic-ai's Ollama provider."""
    if "OLLAMA_BASE_URL" not in os.environ:
        base = settings.ollama.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        os.environ["OLLAMA_BASE_URL"] = base


def create_agent(system: str = AGENT_SYSTEM, cwd: Path | None = None) -> Agent[AgentDeps, str]:
    """Create a pydantic-ai Agent with coding tools."""
    from forge.core.project import build_project_context

    _ensure_ollama_env()

    full_system = system
    if cwd:
        full_system += "\n\n" + build_project_context(cwd)

    return Agent(
        model=f"ollama:{settings.ollama.heavy_model}",
        instructions=full_system,
        tools=ALL_TOOLS,
        deps_type=AgentDeps,
        model_settings=ModelSettings(timeout=300),
    )


async def _save_agent_session(
    db: object,
    session_id: str,
    messages: list[ModelMessage],
) -> None:
    """Persist agent message history as JSON to the conversations table."""
    try:
        # Serialize pydantic-ai messages to JSON
        serialized = []
        for msg in messages:
            if hasattr(msg, "model_dump"):
                serialized.append(msg.model_dump(mode="json"))
            else:
                serialized.append(str(msg))

        history_json = json.dumps(serialized)
        await db.save_message(session_id, "agent_history", history_json, model="")  # type: ignore[union-attr]
    except Exception:
        pass  # Best-effort persistence


async def agent_repl(
    initial_prompt: str | None = None,
    permission: PermissionPolicy | None = None,
    resume_session_id: str | None = None,
    system: str = AGENT_SYSTEM,
) -> None:
    """Run the agentic REPL with tool use."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory

    console = Console()
    cwd = Path.cwd()
    agent = create_agent(system=system, cwd=cwd)
    deps = AgentDeps(
        cwd=cwd,
        console=console,
        permission=permission or PermissionPolicy.AUTO,
    )
    message_history: list[ModelMessage] | None = None
    session: PromptSession[str] = PromptSession(history=InMemoryHistory())

    # DB persistence
    db = await _connect_db()
    session_id = resume_session_id or str(uuid.uuid4())

    if db and not resume_session_id:
        try:
            await db.create_session(session_id, mode="agent" if system is AGENT_SYSTEM else "chat")
        except Exception:
            db = None

    # Show project context info
    from forge.core.project import INSTRUCTION_FILES, detect_project_type

    project_type = detect_project_type(cwd)
    project_info = f"Project: [cyan]{project_type}[/cyan] | " if project_type else ""
    instructions_loaded = any((cwd / f).is_file() for f in INSTRUCTION_FILES)
    instr_info = "[green]instructions loaded[/green]" if instructions_loaded else ""

    persist_info = f"\nSession: [dim]{session_id[:8]}…[/dim]" if db else ""

    console.print(
        Panel(
            f"[bold]Forge v{__version__}[/bold] — {'agentic coding mode' if system is AGENT_SYSTEM else 'chat + tools'}\n"
            f"Model: [green]{settings.ollama.heavy_model}[/green]\n"
            f"Permissions: [{'green' if deps.permission == PermissionPolicy.YOLO else 'yellow'}]"
            f"{deps.permission.value}[/{'green' if deps.permission == PermissionPolicy.YOLO else 'yellow'}]\n"
            f"{project_info}{instr_info}\n"
            f"Working directory: [dim]{deps.cwd}[/dim]"
            f"{persist_info}\n"
            "Type [bold]/help[/bold] for commands, [bold]/quit[/bold] or Ctrl-D to exit",
            title="forge agent" if system is AGENT_SYSTEM else "forge",
            border_style="magenta",
        )
    )

    # Handle initial prompt if provided
    if initial_prompt:
        console.print(f"\n[bold]> {initial_prompt}[/bold]")
        try:
            result = await agent.run(
                initial_prompt,
                deps=deps,
                message_history=message_history,
                event_stream_handler=render_events,
            )
            message_history = result.all_messages()
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
                        message_history = compact_history(message_history)
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
                elif cmd == "/help":
                    console.print(
                        Panel(
                            "/clear    — clear conversation history\n"
                            "/compact  — compact history to fit token budget\n"
                            "/tokens   — show estimated token count\n"
                            "/messages — list message history\n"
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
                message_history = compact_history(message_history)

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

            # Run agent
            try:
                result = await agent.run(
                    user_input,
                    deps=deps,
                    message_history=message_history,
                    event_stream_handler=render_events,
                )
                message_history = result.all_messages()
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
