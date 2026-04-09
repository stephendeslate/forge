"""Command registry — REPL slash-command handlers extracted from loop.py."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from pydantic_ai import Agent, Tool
from pydantic_ai.messages import ModelMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from forge.agent.deps import AgentDeps
from forge.agent.task_store import TaskStore
from forge.config import settings
from forge.log import get_logger

if TYPE_CHECKING:
    from forge.storage.database import Database

logger = get_logger(__name__)


def _suggest_commands(cmd_name: str, commands: dict[str, CommandHandler]) -> str:
    """Suggest similar commands using fuzzy matching."""
    import difflib

    matches = difflib.get_close_matches(cmd_name, commands.keys(), n=3, cutoff=0.6)
    if matches:
        suggestions = ", ".join(matches)
        return f"Did you mean: {suggestions}?"
    return "Type /help for available commands."

_UNCHANGED = object()  # sentinel — None is valid for message_history


@dataclass
class CommandContext:
    """Shared state available to all command handlers."""

    console: Console
    deps: AgentDeps
    agent: Agent
    message_history: list[ModelMessage] | None
    session_id: str
    db: Any  # Database | None
    mcp_servers: list
    rag_available: bool
    rag_project_name: str
    extra_tools: list[Tool]
    system: str
    turn_counter: int


@dataclass
class CommandResult:
    """Return value from command handlers to signal state changes."""

    message_history: object = _UNCHANGED  # list | None | _UNCHANGED
    agent: object = _UNCHANGED  # Agent | _UNCHANGED
    should_break: bool = False
    rag_available: bool | None = None
    turn_counter: int | None = None


# Type alias for command handlers
CommandHandler = Callable[[CommandContext, str], Awaitable[CommandResult]]


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


async def cmd_quit(ctx: CommandContext, args: str) -> CommandResult:
    ctx.console.print("[dim]Goodbye.[/dim]")
    return CommandResult(should_break=True)


async def cmd_clear(ctx: CommandContext, args: str) -> CommandResult:
    ctx.console.print("[dim]Conversation cleared.[/dim]")
    return CommandResult(message_history=None)


async def cmd_compact(ctx: CommandContext, args: str) -> CommandResult:
    from forge.agent.context import count_messages_tokens, smart_compact_history

    if ctx.message_history:
        budget = settings.agent.token_budget
        before = len(ctx.message_history)
        _, before_tokens = count_messages_tokens(ctx.message_history)
        ctx.console.print(
            f"[dim]Compacting {before} messages (~{before_tokens:,} tokens, budget {budget:,})...[/dim]"
        )
        compacted = await smart_compact_history(ctx.message_history, budget)
        after = len(compacted)
        _, after_tokens = count_messages_tokens(compacted)
        ctx.console.print(
            f"[dim]Compacted: {before} → {after} messages "
            f"(~{before_tokens:,} → ~{after_tokens:,} tokens)[/dim]"
        )
        return CommandResult(message_history=compacted)
    else:
        ctx.console.print("[dim]No history to compact.[/dim]")
        return CommandResult()


async def cmd_tokens(ctx: CommandContext, args: str) -> CommandResult:
    from forge.agent.context import count_messages_tokens

    if ctx.message_history:
        budget = settings.agent.token_budget
        count, est_tokens = count_messages_tokens(ctx.message_history)
        parts = [f"{count} messages"]
        # Show real token counts if available
        if ctx.deps.tokens_in > 0:
            parts.append(f"real: {ctx.deps.tokens_in:,} in / {ctx.deps.tokens_out:,} out")
        parts.append(f"est: ~{est_tokens:,} / {budget:,}")
        ctx.console.print(f"[dim]{' | '.join(parts)}[/dim]")
    else:
        ctx.console.print("[dim]No history yet.[/dim]")
    return CommandResult()


async def cmd_messages(ctx: CommandContext, args: str) -> CommandResult:
    if ctx.message_history:
        for i, msg in enumerate(ctx.message_history):
            kind = type(msg).__name__
            text_len = len(str(msg))
            ctx.console.print(f"[dim]  {i}: {kind} ({text_len} chars)[/dim]")
    else:
        ctx.console.print("[dim]No history yet.[/dim]")
    return CommandResult()


async def cmd_status(ctx: CommandContext, args: str) -> CommandResult:
    ctx.deps.status_visible = not ctx.deps.status_visible
    if ctx.deps.status_tracker:
        ctx.deps.status_tracker.visible = ctx.deps.status_visible
        if not ctx.deps.status_visible:
            ctx.deps.status_tracker._clear_line()
    state = "visible" if ctx.deps.status_visible else "hidden"
    ctx.console.print(f"[dim]Status line: {state}[/dim]")
    return CommandResult()


async def cmd_think(ctx: CommandContext, args: str) -> CommandResult:
    ctx.deps.thinking_enabled = not ctx.deps.thinking_enabled
    state = "on" if ctx.deps.thinking_enabled else "off"
    ctx.console.print(f"[dim]Extended thinking: {state}[/dim]")
    return CommandResult()


async def cmd_cloud(ctx: CommandContext, args: str) -> CommandResult:
    """Toggle cloud-based reasoning (Gemini 2.5 Pro)."""
    import os

    api_key = settings.gemini.api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key and not ctx.deps.cloud_reasoning_enabled:
        ctx.console.print(
            "[red]No API key configured.[/red] Set GOOGLE_API_KEY env var "
            "or add api_key to [gemini] in config.toml"
        )
        return CommandResult()

    ctx.deps.cloud_reasoning_enabled = not ctx.deps.cloud_reasoning_enabled
    state = "on" if ctx.deps.cloud_reasoning_enabled else "off"
    if ctx.deps.cloud_reasoning_enabled:
        ctx.console.print(
            f"[yellow]Cloud reasoning: {state}[/yellow]\n"
            f"[dim]Model: {settings.gemini.model} (fallback: {settings.gemini.fallback_model}) | "
            f"Used for: /plan, circuit-breaker escalation\n"
            f"Warning: Prompts and code context will be sent to Google's API[/dim]"
        )
    else:
        ctx.console.print(f"[dim]Cloud reasoning: {state} — all processing is local[/dim]")
    return CommandResult()


async def cmd_plan(ctx: CommandContext, args: str) -> CommandResult:
    if not args.strip():
        ctx.console.print("[dim]Usage: /plan <prompt>[/dim]")
        return CommandResult()

    from forge.agent.loop import _plan_and_execute

    try:
        result = await _plan_and_execute(ctx.agent, args.strip(), ctx.deps, ctx.message_history)
        if result is not None:
            return CommandResult(message_history=result)
    except KeyboardInterrupt:
        ctx.console.print("\n[dim]Interrupted.[/dim]")
    except Exception as e:
        from forge.agent.loop import _handle_agent_error
        _handle_agent_error(ctx.console, e)
    return CommandResult()


async def cmd_tools(ctx: CommandContext, args: str) -> CommandResult:
    ctx.deps.tools_visible = not ctx.deps.tools_visible
    if ctx.deps.turn_buffer and ctx.deps.turn_buffer._items:
        ctx.deps.turn_buffer.rerender(ctx.deps.tools_visible)
    else:
        state = "visible" if ctx.deps.tools_visible else "hidden"
        ctx.console.print(f"[dim]Tool results: {state}[/dim]")
    return CommandResult()


async def cmd_model(ctx: CommandContext, args: str) -> CommandResult:
    if not args.strip():
        current = ctx.deps.model_override or settings.ollama.heavy_model
        ctx.console.print(
            f"[dim]Current model: {current}\n"
            f"Usage: /model fast | /model heavy | /model opus | /model <name>[/dim]"
        )
        return CommandResult()

    arg = args.strip()
    if arg == "heavy":
        ctx.deps.model_override = None
        ctx.console.print(f"[dim]Model: {settings.ollama.heavy_model} (heavy/local)[/dim]")
    elif arg == "fast":
        ctx.deps.model_override = settings.ollama.fast_model
        if ctx.deps.escalator:
            ctx.deps.escalator.reset()
        ctx.console.print(f"[dim]Model: {settings.ollama.fast_model} (fast)[/dim]")
    else:
        ctx.deps.model_override = arg
        ctx.console.print(f"[dim]Model: {arg}[/dim]")
    return CommandResult()


async def cmd_plan_status(ctx: CommandContext, args: str) -> CommandResult:
    if ctx.deps.active_plan:
        ctx.console.print(
            Panel(ctx.deps.active_plan, title="Active Plan", border_style="cyan")
        )
    else:
        ctx.console.print("[dim]No active plan.[/dim]")
    return CommandResult()


async def cmd_tasks(ctx: CommandContext, args: str) -> CommandResult:
    if ctx.deps.task_store:
        tasks = ctx.deps.task_store.list_all()
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
                table.add_row(
                    t.id,
                    f"[{status_style}]{t.status.value}[/{status_style}]",
                    t.subject,
                    blocked,
                )
            ctx.console.print(table)
        else:
            ctx.console.print("[dim]No tasks.[/dim]")
    else:
        ctx.console.print("[dim]Task tracking unavailable.[/dim]")
    return CommandResult()


async def cmd_memory(ctx: CommandContext, args: str) -> CommandResult:
    parts = args.strip().split(maxsplit=1)
    if parts and parts[0].lower() == "search" and len(parts) > 1:
        query = parts[1]
        if ctx.deps.memory_db and ctx.deps.memory_project:
            try:
                from forge.agent.memory import recall_from_db

                rows = await recall_from_db(
                    ctx.deps.memory_db, ctx.deps.memory_project, query,
                    min_score=0.1,  # Permissive for explicit user searches
                )
                if rows:
                    for r in rows:
                        score = f"{r.score:.2f}" if r.score else ""
                        ctx.console.print(
                            f"  [cyan]#{r.id}[/cyan] [{r.category}] "
                            f"[bold]{r.subject}[/bold]: {r.content} [dim]{score}[/dim]"
                        )
                else:
                    ctx.console.print("[dim]No matching memories.[/dim]")
            except Exception as e:
                ctx.console.print(f"[red]Memory search error:[/red] {e}")
        else:
            ctx.console.print("[dim]Memory unavailable (no database).[/dim]")
    else:
        if ctx.deps.memory_db and ctx.deps.memory_project:
            try:
                count = await ctx.deps.memory_db.count_memories(ctx.deps.memory_project)
                ctx.console.print(
                    f"[dim]{count} memories for project '{ctx.deps.memory_project}'[/dim]"
                )
                ctx.console.print("[dim]Usage: /memory search <query>[/dim]")
            except Exception:
                ctx.console.print("[dim]Memory stats unavailable.[/dim]")
        else:
            ctx.console.print("[dim]Memory unavailable (no database).[/dim]")
    return CommandResult()


async def cmd_forget(ctx: CommandContext, args: str) -> CommandResult:
    if not args.strip():
        ctx.console.print("[dim]Usage: /forget <memory_id>[/dim]")
        return CommandResult()
    try:
        mid = int(args.strip())
        if ctx.deps.memory_db:
            deleted = await ctx.deps.memory_db.delete_memory(mid)
            if deleted:
                ctx.console.print(f"[dim]Deleted memory #{mid}.[/dim]")
            else:
                ctx.console.print(f"[dim]Memory #{mid} not found.[/dim]")
        else:
            ctx.console.print("[dim]Memory unavailable (no database).[/dim]")
    except ValueError:
        ctx.console.print("[dim]Usage: /forget <memory_id> (numeric ID)[/dim]")
    return CommandResult()


async def cmd_mcp(ctx: CommandContext, args: str) -> CommandResult:
    if ctx.mcp_servers:
        for server in ctx.mcp_servers:
            sid = getattr(server, "id", None) or "unknown"
            stype = type(server).__name__
            is_up = getattr(server, "is_running", False)
            state = "running" if is_up else "stopped"
            ctx.console.print(f"  [cyan]{sid}[/cyan] — {stype} [{state}]")
    else:
        ctx.console.print("[dim]No MCP servers configured.[/dim]")
    return CommandResult()


async def cmd_help(ctx: CommandContext, args: str) -> CommandResult:
    ctx.console.print(
        Panel(
            "/clear         — clear conversation history\n"
            "/compact       — compact history to fit token budget\n"
            "/tokens        — show estimated token count\n"
            "/messages      — list message history\n"
            "/model         — switch model (fast/heavy/opus/<name>)\n"
            "/status        — toggle status line (or Ctrl-O)\n"
            "/tools         — toggle tool result display (or Ctrl-R)\n"
            "/think         — toggle extended thinking on/off\n"
            "/cloud         — toggle cloud reasoning (Gemini 2.5 Pro)\n"
            "/plan          — plan before executing (e.g. /plan refactor X)\n"
            "/plan-status   — show active plan\n"
            "/tasks         — show task list\n"
            "/mcp           — list connected MCP servers\n"
            "/memory        — show memory stats / search memories\n"
            "/forget <id>   — delete a memory by ID\n"
            "/exemplars     — list captured cloud model exemplars\n"
            "/checkpoint    — save conversation checkpoint [name]\n"
            "/restore <n>   — restore to named checkpoint\n"
            "/checkpoints   — list saved checkpoints\n"
            "/index         — index/reindex project for RAG\n"
            "/cwd           — show working directory\n"
            "/cd <dir>      — change working directory\n"
            "/worktree      — create isolated git worktree\n"
            "/quit          — exit",
            title="commands",
            border_style="dim",
        )
    )
    return CommandResult()


async def cmd_cwd(ctx: CommandContext, args: str) -> CommandResult:
    ctx.console.print(f"[dim]{ctx.deps.cwd}[/dim]")
    return CommandResult()


async def cmd_cd(ctx: CommandContext, args: str) -> CommandResult:
    if not args.strip():
        ctx.console.print("[dim]Usage: /cd <directory>[/dim]")
        return CommandResult()

    from forge.agent.loop import _rebuild_agent

    new_dir = Path(args.strip()).expanduser()
    if not new_dir.is_absolute():
        new_dir = ctx.deps.cwd / new_dir
    new_dir = new_dir.resolve()

    if new_dir.is_dir():
        ctx.deps.cwd = new_dir
        new_agent = _rebuild_agent(
            ctx.deps,
            ctx.system,
            ctx.extra_tools,
            toolsets=ctx.mcp_servers or None,
        )
        ctx.console.print(f"[dim]Working directory: {ctx.deps.cwd} (agent reloaded)[/dim]")
        return CommandResult(agent=new_agent)
    else:
        ctx.console.print(f"[red]Not a directory: {new_dir}[/red]")
        return CommandResult()


async def cmd_worktree(ctx: CommandContext, args: str) -> CommandResult:
    if ctx.deps.worktree:
        ctx.console.print("[yellow]Already in a worktree.[/yellow]")
        return CommandResult()

    from forge.agent.loop import _rebuild_agent
    from forge.agent.worktree import create_worktree, is_git_repo

    if not is_git_repo(ctx.deps.cwd):
        ctx.console.print("[red]Not a git repository — cannot create worktree.[/red]")
        return CommandResult()

    wt_name = args.strip() or None
    try:
        wt_info = create_worktree(ctx.deps.cwd, wt_name)
        ctx.deps.worktree = wt_info
        ctx.deps.cwd = wt_info.path
        new_agent = _rebuild_agent(
            ctx.deps,
            ctx.system,
            ctx.extra_tools,
            toolsets=ctx.mcp_servers or None,
        )
        ctx.console.print(
            f"[green]Worktree created:[/green] {wt_info.path}\n"
            f"[green]Branch:[/green] {wt_info.branch}"
        )
        return CommandResult(agent=new_agent)
    except RuntimeError as e:
        ctx.console.print(f"[red]Worktree error:[/red] {e}")
        return CommandResult()


async def cmd_checkpoint(ctx: CommandContext, args: str) -> CommandResult:
    if not ctx.db:
        ctx.console.print("[yellow]Checkpoints require persistence (database).[/yellow]")
        return CommandResult()

    from forge.agent.loop import _message_list_adapter

    cp_name = args.strip() or f"cp-{ctx.turn_counter}"
    if not ctx.message_history:
        ctx.console.print("[dim]No history to checkpoint.[/dim]")
        return CommandResult()

    try:
        history_json = _message_list_adapter.dump_json(ctx.message_history).decode()
        task_json = ctx.deps.task_store.to_json() if ctx.deps.task_store else None
        await ctx.db.save_checkpoint(
            ctx.session_id, cp_name, history_json, task_json, len(ctx.message_history)
        )
        ctx.console.print(
            f"[green]Checkpoint '{cp_name}' saved ({len(ctx.message_history)} messages).[/green]"
        )
    except Exception as e:
        ctx.console.print(f"[red]Checkpoint save failed:[/red] {e}")
    return CommandResult()


async def cmd_restore(ctx: CommandContext, args: str) -> CommandResult:
    if not ctx.db:
        ctx.console.print("[yellow]Checkpoints require persistence (database).[/yellow]")
        return CommandResult()

    from forge.agent.loop import _message_list_adapter

    if not args.strip():
        ctx.console.print("[dim]Usage: /restore <name>[/dim]")
        return CommandResult()

    cp_name = args.strip()
    try:
        cp = await ctx.db.load_checkpoint(ctx.session_id, cp_name)
        if not cp:
            ctx.console.print(f"[yellow]Checkpoint '{cp_name}' not found.[/yellow]")
            return CommandResult()
        restored = _message_list_adapter.validate_json(cp["agent_history"])
        if cp.get("task_store"):
            ctx.deps.task_store = TaskStore.from_json(cp["task_store"])
        ctx.console.print(
            f"[green]Restored to checkpoint '{cp_name}' ({cp['message_count']} messages).[/green]"
        )
        return CommandResult(message_history=restored)
    except Exception as e:
        ctx.console.print(f"[red]Restore failed:[/red] {e}")
        return CommandResult()


async def cmd_checkpoints(ctx: CommandContext, args: str) -> CommandResult:
    if not ctx.db:
        ctx.console.print("[yellow]Checkpoints require persistence (database).[/yellow]")
        return CommandResult()
    try:
        cps = await ctx.db.list_checkpoints(ctx.session_id)
        if cps:
            from rich.table import Table

            table = Table(title="Checkpoints", border_style="dim")
            table.add_column("Name", style="cyan")
            table.add_column("Messages", justify="right")
            table.add_column("Created", style="dim")
            for cp in cps:
                table.add_row(
                    cp["name"],
                    str(cp["message_count"]),
                    str(cp["created_at"].strftime("%Y-%m-%d %H:%M")),
                )
            ctx.console.print(table)
        else:
            ctx.console.print("[dim]No checkpoints. Use /checkpoint [name] to save one.[/dim]")
    except Exception as e:
        ctx.console.print(f"[red]Checkpoint list failed:[/red] {e}")
    return CommandResult()


async def cmd_exemplars(ctx: CommandContext, args: str) -> CommandResult:
    """List or manage exemplars — captured cloud model successes for local model learning."""
    parts = args.strip().split(maxsplit=1)

    # /exemplars show <id>
    if parts and parts[0].lower() == "show" and len(parts) > 1:
        try:
            eid = int(parts[1])
            if ctx.deps.memory_db:
                ex = await ctx.deps.memory_db.get_exemplar(eid)
                if ex:
                    created = ex.created_at.strftime("%Y-%m-%d %H:%M") if ex.created_at else "?"
                    last_used = ex.last_used_at.strftime("%Y-%m-%d %H:%M") if ex.last_used_at else "never"
                    ctx.console.print(Panel(
                        f"[cyan]Type:[/cyan] {ex.task_type}  "
                        f"[cyan]Source:[/cyan] {ex.model_source}  "
                        f"[cyan]Score:[/cyan] {ex.outcome_score:.2f}  "
                        f"[cyan]Used:[/cyan] {ex.used_count}x\n"
                        f"[cyan]Created:[/cyan] {created}  "
                        f"[cyan]Last used:[/cyan] {last_used}\n\n"
                        f"[bold]Task:[/bold]\n{ex.task_description}\n\n"
                        f"[bold]Approach:[/bold]\n{ex.solution_approach[:2000]}"
                        + (f"\n... ({len(ex.solution_approach)} chars total)" if len(ex.solution_approach) > 2000 else ""),
                        title=f"Exemplar #{ex.id}",
                        border_style="cyan",
                    ))
                else:
                    ctx.console.print(f"[dim]Exemplar #{eid} not found.[/dim]")
            else:
                ctx.console.print("[dim]Database unavailable.[/dim]")
        except ValueError:
            ctx.console.print("[dim]Usage: /exemplars show <id> (numeric ID)[/dim]")
        return CommandResult()

    # /exemplars delete <id>
    if parts and parts[0].lower() == "delete" and len(parts) > 1:
        try:
            eid = int(parts[1])
            if ctx.deps.memory_db:
                deleted = await ctx.deps.memory_db.delete_exemplar(eid)
                if deleted:
                    ctx.console.print(f"[dim]Deleted exemplar #{eid}.[/dim]")
                else:
                    ctx.console.print(f"[dim]Exemplar #{eid} not found.[/dim]")
            else:
                ctx.console.print("[dim]Database unavailable.[/dim]")
        except ValueError:
            ctx.console.print("[dim]Usage: /exemplars delete <id> (numeric ID)[/dim]")
        return CommandResult()

    # /exemplars — list all
    if ctx.deps.memory_db and ctx.deps.memory_project:
        try:
            exemplars = await ctx.deps.memory_db.list_exemplars(ctx.deps.memory_project)
            if exemplars:
                from rich.table import Table

                table = Table(title="Exemplars", border_style="dim")
                table.add_column("ID", style="cyan", justify="right")
                table.add_column("Type")
                table.add_column("Task", max_width=40)
                table.add_column("Score", justify="right")
                table.add_column("Source")
                table.add_column("Used", justify="right")
                for ex in exemplars:
                    score_style = "green" if ex.outcome_score >= 0.7 else "yellow" if ex.outcome_score >= 0.4 else "red"
                    table.add_row(
                        str(ex.id),
                        ex.task_type,
                        ex.task_description[:40] + ("..." if len(ex.task_description) > 40 else ""),
                        f"[{score_style}]{ex.outcome_score:.2f}[/{score_style}]",
                        ex.model_source,
                        str(ex.used_count),
                    )
                ctx.console.print(table)
                ctx.console.print("[dim]/exemplars show <id> — view details  |  /exemplars delete <id> — remove[/dim]")
            else:
                ctx.console.print(
                    "[dim]No exemplars yet. They're captured automatically when "
                    "cloud models (Gemini) succeed at recovery, planning, or critique.[/dim]"
                )
        except Exception as e:
            ctx.console.print(f"[red]Exemplar list error:[/red] {e}")
    else:
        ctx.console.print("[dim]Database unavailable — exemplars require PostgreSQL.[/dim]")
    return CommandResult()


async def cmd_index(ctx: CommandContext, args: str) -> CommandResult:
    if not ctx.db:
        ctx.console.print("[yellow]Indexing requires a database.[/yellow]")
        return CommandResult()

    from forge.agent.loop import _rebuild_agent

    try:
        from forge.rag.indexer import index_directory

        stats = await index_directory(ctx.deps.cwd, ctx.db, project=ctx.rag_project_name)
        ctx.console.print(
            f"[green]Indexed:[/green] {stats['files_indexed']} files, "
            f"{stats['chunks_stored']} chunks "
            f"({stats['files_skipped']} skipped)"
        )
        if not ctx.rag_available:
            from forge.agent.tools import rag_search

            ctx.deps.rag_db = ctx.db
            ctx.deps.rag_project = ctx.rag_project_name
            ctx.extra_tools.append(rag_search)
            new_agent = _rebuild_agent(
                ctx.deps,
                ctx.system,
                ctx.extra_tools,
                toolsets=ctx.mcp_servers or None,
            )
            ctx.console.print("[green]RAG search enabled.[/green]")
            return CommandResult(agent=new_agent, rag_available=True)
    except Exception as e:
        ctx.console.print(f"[red]Indexing failed:[/red] {e}")
    return CommandResult()


# ---------------------------------------------------------------------------
# Command registry
# ---------------------------------------------------------------------------

COMMANDS: dict[str, CommandHandler] = {
    "/quit": cmd_quit,
    "/exit": cmd_quit,
    "/q": cmd_quit,
    "/clear": cmd_clear,
    "/compact": cmd_compact,
    "/tokens": cmd_tokens,
    "/messages": cmd_messages,
    "/status": cmd_status,
    "/think": cmd_think,
    "/cloud": cmd_cloud,
    "/plan": cmd_plan,
    "/tools": cmd_tools,
    "/model": cmd_model,
    "/plan-status": cmd_plan_status,
    "/tasks": cmd_tasks,
    "/memory": cmd_memory,
    "/forget": cmd_forget,
    "/exemplars": cmd_exemplars,
    "/mcp": cmd_mcp,
    "/help": cmd_help,
    "/cwd": cmd_cwd,
    "/cd": cmd_cd,
    "/worktree": cmd_worktree,
    "/checkpoint": cmd_checkpoint,
    "/restore": cmd_restore,
    "/checkpoints": cmd_checkpoints,
    "/index": cmd_index,
}


async def dispatch(cmd_ctx: CommandContext, user_input: str) -> CommandResult | None:
    """Dispatch a slash command. Returns CommandResult if handled, None if not a command."""
    if not user_input.startswith("/"):
        return None

    parts = user_input.split(maxsplit=1)
    cmd_name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    handler = COMMANDS.get(cmd_name)
    if handler is None:
        hint = _suggest_commands(cmd_name, COMMANDS)
        cmd_ctx.console.print(f"[dim]Unknown command: {cmd_name}. {hint}[/dim]")
        return CommandResult()  # handled (as unknown), but no state change

    return await handler(cmd_ctx, args)
