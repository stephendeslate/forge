"""Forge CLI — REPL + commands via typer."""

from __future__ import annotations

import asyncio
import uuid

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from forge import __version__
from forge.config import settings
from forge.core.conversation import Conversation
from forge.core.router import ModelRouter, Route
from forge.models.npu import get_npu_backend
from forge.models.ollama import get_fast_backend, get_heavy_backend
from forge.prompts.system import CHAT_SYSTEM, CODE_SYSTEM
from forge.rag.retriever import format_context, retrieve
from forge.storage.database import Database

app = typer.Typer(
    name="forge",
    help="Local AI orchestration — smart routing, RAG, and self-correction.",
    no_args_is_help=False,
    invoke_without_command=True,
)
console = Console()


def _get_router() -> ModelRouter:
    return ModelRouter(heavy=get_heavy_backend(), fast=get_fast_backend(), npu=get_npu_backend())


def _resolve_route(gpu: bool, fast: bool, npu: bool = False) -> Route | None:
    if npu:
        return Route.NPU
    if gpu:
        return Route.HEAVY
    if fast:
        return Route.FAST
    return None


def _handle_model_error(e: Exception, backend_name: str) -> None:
    """Print a user-friendly error message for model failures."""
    err_str = str(e).lower()
    if "connection" in err_str or "connect" in err_str:
        console.print(
            f"[red]Cannot connect to Ollama.[/red] Is it running? "
            f"Check with: [bold]systemctl status ollama[/bold]"
        )
    elif "timeout" in err_str or "timed out" in err_str:
        console.print(
            f"[red]Request timed out[/red] ({backend_name}). "
            f"The model may still be loading — try again in a moment."
        )
    elif "404" in err_str:
        console.print(
            f"[red]Model not found[/red] ({backend_name}). "
            f"Pull it with: [bold]ollama pull <model>[/bold]"
        )
    else:
        console.print(f"[red]Error ({backend_name}):[/red] {e}")


async def _stream_response(
    router: ModelRouter,
    prompt: str,
    *,
    system: str = CHAT_SYSTEM,
    force_route: Route | None = None,
    history: str = "",
) -> tuple[str, str]:
    """Stream a response and return (full_text, model_used)."""
    route, backend = router.route(prompt, force=force_route)

    full_prompt = prompt
    if history:
        full_prompt = f"Conversation so far:\n{history}\n\nUser: {prompt}"

    collected: list[str] = []

    console.print(f"[dim]({backend.name} → {backend.model_id})[/dim]")

    try:
        with Live(console=console, refresh_per_second=12, vertical_overflow="visible") as live:
            async for chunk in backend.stream(full_prompt, system=system):
                collected.append(chunk)
                text = "".join(collected)
                live.update(Markdown(text))
    except Exception as e:
        _handle_model_error(e, backend.name)
        return "", backend.model_id

    full_text = "".join(collected)
    return full_text, backend.model_id


async def _generate_response(
    router: ModelRouter,
    prompt: str,
    *,
    system: str = CHAT_SYSTEM,
    force_route: Route | None = None,
    history: str = "",
) -> tuple[str, str]:
    """Generate a complete response (no streaming)."""
    route, backend = router.route(prompt, force=force_route)

    full_prompt = prompt
    if history:
        full_prompt = f"Conversation so far:\n{history}\n\nUser: {prompt}"

    console.print(f"[dim]({backend.name} → {backend.model_id})[/dim]")
    try:
        result = await backend.generate(full_prompt, system=system)
    except Exception as e:
        _handle_model_error(e, backend.name)
        return "", backend.model_id
    return result, backend.model_id


async def _one_shot(
    prompt: str,
    *,
    system: str = CHAT_SYSTEM,
    force_route: Route | None = None,
    stream: bool = True,
) -> None:
    """Handle a single prompt → response."""
    from pathlib import Path

    from forge.core.project import build_project_context

    # Make one-shot aware of the current working directory
    project_ctx = build_project_context(Path.cwd())
    if project_ctx:
        system = system + "\n\n" + project_ctx

    router = _get_router()

    if stream and settings.streaming:
        await _stream_response(router, prompt, system=system, force_route=force_route)
    else:
        text, model = await _generate_response(
            router, prompt, system=system, force_route=force_route
        )
        if text:
            console.print(Markdown(text))





@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
) -> None:
    """Forge — local AI orchestration."""
    if version:
        console.print(f"forge {__version__}")
        raise typer.Exit()

    # If no subcommand, launch tool-using REPL with chat persona
    if ctx.invoked_subcommand is None:
        from forge.agent.loop import agent_repl

        asyncio.run(agent_repl(system=CHAT_SYSTEM))


@app.command()
def agent(
    prompt: str = typer.Argument(None, help="Optional initial prompt"),
    yolo: bool = typer.Option(False, "--yolo", help="Allow all tool calls without prompting"),
    ask: bool = typer.Option(False, "--ask", help="Prompt for every tool call"),
) -> None:
    """Agentic coding mode — read, edit, search, and run commands."""
    from forge.agent.loop import agent_repl
    from forge.agent.permissions import PermissionPolicy

    if yolo:
        policy = PermissionPolicy.YOLO
    elif ask:
        policy = PermissionPolicy.ASK
    else:
        policy = PermissionPolicy.AUTO

    asyncio.run(agent_repl(prompt, permission=policy))


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Question or prompt"),
    gpu: bool = typer.Option(False, "--gpu", help="Force heavy GPU model"),
    fast: bool = typer.Option(False, "--fast", help="Force fast GPU model"),
    npu: bool = typer.Option(False, "--npu", help="Force NPU model"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming"),
) -> None:
    """One-shot query — ask a question and get a response."""
    force = _resolve_route(gpu, fast, npu)
    asyncio.run(_one_shot(prompt, force_route=force, stream=not no_stream))


@app.command()
def code(
    prompt: str = typer.Argument(None, help="Optional initial prompt"),
    gpu: bool = typer.Option(False, "--gpu", help="Force heavy GPU model"),
    fast: bool = typer.Option(False, "--fast", help="Force fast GPU model"),
    npu: bool = typer.Option(False, "--npu", help="Force NPU model"),
    project: str = typer.Option(None, "--project", "-p", help="Project name (defaults to cwd name)"),
) -> None:
    """REPL with codebase RAG context from indexed project."""
    asyncio.run(_code_command(prompt, gpu, fast, npu, project))


async def _code_command(
    prompt: str | None, gpu: bool, fast: bool, npu: bool, project: str | None
) -> None:
    import os

    proj = project or os.path.basename(os.getcwd())
    db = Database()

    try:
        await db.connect()
    except Exception as e:
        console.print(f"[yellow]Database unavailable — running without RAG.[/yellow] ({e})")
        db = None  # type: ignore[assignment]

    if prompt:
        force = _resolve_route(gpu, fast, npu)
        system = CODE_SYSTEM
        if db:
            chunks = await retrieve(prompt, proj, db)
            if chunks:
                ctx = format_context(chunks)
                system = f"{CODE_SYSTEM}\n\n{ctx}"
                console.print(f"[dim]Retrieved {len(chunks)} chunks from project '{proj}'[/dim]")
        router = _get_router()
        if settings.streaming:
            await _stream_response(router, prompt, system=system, force_route=force)
        else:
            text, _ = await _generate_response(router, prompt, system=system, force_route=force)
            if text:
                console.print(Markdown(text))
    else:
        await _code_repl(db, proj)

    if db:
        await db.close()


async def _code_repl(
    db: Database | None,
    project: str,
    resume_session_id: str | None = None,
) -> None:
    """Interactive REPL with RAG context injection."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory

    router = _get_router()
    session: PromptSession[str] = PromptSession(history=InMemoryHistory())

    # Session persistence (reuse existing db connection for both RAG and persistence)
    persist_db = db if settings.persist_history else None
    session_id = resume_session_id or str(uuid.uuid4())

    if persist_db and not resume_session_id:
        try:
            await persist_db.create_session(session_id, mode="code", project=project)
        except Exception:
            persist_db = None

    if resume_session_id and persist_db:
        conversation = await Conversation.load_from_db(
            session_id, persist_db, max_turns=settings.max_history
        )
        console.print(f"[dim]Resumed session {session_id[:8]}… ({conversation.turn_count} turns)[/dim]")
    else:
        conversation = Conversation(
            max_turns=settings.max_history, session_id=session_id, db=persist_db
        )

    rag_status = f"[green]RAG: {project}[/green]" if db else "[yellow]RAG: offline[/yellow]"
    persist_info = f"Session: [dim]{session_id[:8]}…[/dim]" if persist_db else ""
    console.print(
        Panel(
            f"[bold]Forge v{__version__}[/bold] — code mode\n"
            f"Models: [green]{settings.ollama.heavy_model}[/green] (heavy) "
            f"+ [cyan]{settings.ollama.fast_model}[/cyan] (fast)\n"
            f"{rag_status}\n"
            + (f"{persist_info}\n" if persist_info else "")
            + "Type [bold]/help[/bold] for commands, [bold]/quit[/bold] or Ctrl-D to exit",
            title="forge code",
            border_style="green",
        )
    )

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

        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "/q"):
                break
            elif cmd == "/clear":
                conversation.clear()
                console.print("[dim]Conversation cleared.[/dim]")
                continue
            elif cmd == "/sessions":
                if persist_db:
                    await _show_sessions(persist_db)
                else:
                    console.print("[dim]Persistence disabled.[/dim]")
                continue
            elif cmd == "/help":
                console.print(Panel(
                    "/clear    — clear conversation\n"
                    "/sessions — list recent sessions\n"
                    "/heavy    — force next response to use heavy model\n"
                    "/fast     — force next response to use fast model\n"
                    "/npu      — force next response to use NPU model\n"
                    "/quit     — exit",
                    title="commands", border_style="dim",
                ))
                continue
            elif cmd in ("/heavy", "/fast", "/npu"):
                force = {"/heavy": Route.HEAVY, "/fast": Route.FAST, "/npu": Route.NPU}[cmd]
                console.print(f"[dim]Next response will use {force.value} model.[/dim]")
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

                # Retrieve RAG context for forced route
                code_system = CODE_SYSTEM
                if db:
                    try:
                        chunks = await retrieve(user_input, project, db)
                        if chunks:
                            ctx = format_context(chunks)
                            code_system = f"{CODE_SYSTEM}\n\n{ctx}"
                            console.print(f"[dim]Retrieved {len(chunks)} chunks[/dim]")
                    except Exception as e:
                        console.print(f"[dim]RAG error: {e}[/dim]")

                conversation.add("user", user_input)
                history = conversation.format_history()
                text, model = await _stream_response(
                    router, user_input, system=code_system, force_route=force, history=history
                )
                if text:
                    conversation.add("assistant", text, model=model)
                continue
            else:
                console.print(f"[dim]Unknown command: {cmd}[/dim]")
                continue

        # Retrieve RAG context
        system = CODE_SYSTEM
        if db:
            try:
                chunks = await retrieve(user_input, project, db)
                if chunks:
                    ctx = format_context(chunks)
                    system = f"{CODE_SYSTEM}\n\n{ctx}"
                    console.print(f"[dim]Retrieved {len(chunks)} chunks[/dim]")
            except Exception as e:
                console.print(f"[dim]RAG error: {e}[/dim]")

        conversation.add("user", user_input)
        history = conversation.format_history()
        text, model = await _stream_response(
            router, user_input, system=system, history=history
        )
        if text:
            conversation.add("assistant", text, model=model)


@app.command()
def status() -> None:
    """Show model status and configuration."""
    import httpx

    console.print(Panel("[bold]Forge Status[/bold]", border_style="blue"))

    # Check Ollama
    try:
        resp = httpx.get(f"{settings.ollama.base_url}/api/tags", timeout=5)
        models = resp.json().get("models", [])
        model_names = [m["name"] for m in models]

        console.print(f"\n[bold]Ollama[/bold] ({settings.ollama.base_url})")
        heavy = settings.ollama.heavy_model
        fast = settings.ollama.fast_model
        embed = settings.ollama.embed_model

        for name, label in [(heavy, "heavy"), (fast, "fast"), (embed, "embed")]:
            available = any(name in m for m in model_names)
            icon = "[green]●[/green]" if available else "[red]○[/red]"
            console.print(f"  {icon} {name} ({label})")

        # Check GPU backend from running models
        try:
            ps_resp = httpx.get(f"{settings.ollama.base_url}/api/ps", timeout=5)
            running = ps_resp.json().get("models", [])
            if running:
                console.print(f"\n  [bold]Running:[/bold]")
                for m in running:
                    size_gb = m.get("size_vram", 0) / (1024**3)
                    console.print(f"    {m['name']} ({size_gb:.1f} GiB VRAM)")
        except Exception:
            pass

    except httpx.ConnectError:
        console.print("[red]Ollama unreachable[/red] — is it running?")
    except Exception as e:
        console.print(f"[red]Ollama error:[/red] {e}")

    # NPU status
    console.print(f"\n[bold]NPU[/bold]")
    if settings.npu.enabled:
        try:
            npu_resp = httpx.get(f"{settings.npu.base_url.rstrip('/')}/models", timeout=5)
            npu_models = npu_resp.json().get("data", [])
            model_names = [m.get("id", "unknown") for m in npu_models]
            console.print(f"  [green]●[/green] Connected — {', '.join(model_names) or 'no models'}")
        except Exception:
            console.print(f"  [yellow]●[/yellow] Enabled but unreachable ({settings.npu.base_url})")
    else:
        console.print("  [dim]Disabled (set FORGE_NPU_ENABLED=true)[/dim]")

    # Database / RAG status
    console.print(f"\n[bold]Database[/bold]")
    try:
        db = Database()
        asyncio.run(_check_db_status(db))
    except Exception as e:
        console.print(f"  [red]Unavailable:[/red] {e}")

    console.print(f"\n[bold]Config[/bold]")
    console.print(f"  Default route: {settings.default_route}")
    console.print(f"  Streaming: {settings.streaming}")
    console.print(f"  Max history: {settings.max_history} turns")


async def _check_db_status(db: Database) -> None:
    try:
        await db.connect()
        count = await db.get_session_count()
        console.print(f"  [green]●[/green] Connected — {count} sessions")
        await db.close()
    except Exception:
        console.print("  [red]○[/red] Cannot connect — is PostgreSQL running?")


def _format_age(dt: object) -> str:
    """Format a datetime as a human-readable age string."""
    from datetime import datetime, timezone

    if not dt:
        return "?"
    if hasattr(dt, "tzinfo") and getattr(dt, "tzinfo", None) is not None:
        now = datetime.now(timezone.utc)
    else:
        now = datetime.now()
    delta = now - dt  # type: ignore[operator]
    secs = int(delta.total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    days = secs // 86400
    if days == 1:
        return "yesterday"
    return f"{days}d ago"


async def _show_sessions(db: Database, limit: int = 20) -> None:
    """Display a table of recent sessions."""
    sessions = await db.list_sessions(limit=limit)
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Recent Sessions", border_style="dim")
    table.add_column("ID", style="cyan", max_width=8)
    table.add_column("Title", max_width=40)
    table.add_column("Mode", style="green")
    table.add_column("Turns", justify="right")
    table.add_column("Age", style="dim")

    for s in sessions:
        turns = s["message_count"] // 2  # messages → turns
        table.add_row(
            s["id"][:8],
            s["title"] or "[dim]untitled[/dim]",
            s["mode"],
            str(turns),
            _format_age(s["updated_at"]),
        )

    console.print(table)


@app.command()
def history(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of sessions to show"),
) -> None:
    """List recent conversation sessions."""
    asyncio.run(_history_command(limit))


async def _history_command(limit: int) -> None:
    db = Database()
    try:
        await db.connect()
    except Exception as e:
        console.print(f"[red]Cannot connect to database:[/red] {e}")
        raise typer.Exit(1)

    try:
        await _show_sessions(db, limit=limit)
    finally:
        await db.close()


@app.command()
def resume(
    session_id: str = typer.Argument(None, help="Session ID to resume (default: most recent)"),
) -> None:
    """Resume a previous conversation session."""
    asyncio.run(_resume_command(session_id))


async def _resume_command(session_id: str | None) -> None:
    db = Database()
    try:
        await db.connect()
    except Exception as e:
        console.print(f"[red]Cannot connect to database:[/red] {e}")
        raise typer.Exit(1)

    try:
        # Resolve session ID
        if not session_id:
            sid = await db.get_latest_session_id()
            if not sid:
                console.print("[dim]No sessions found.[/dim]")
                raise typer.Exit(0)
            session_id = sid
        else:
            # Allow short prefix matching
            sessions = await db.list_sessions(limit=100)
            matches = [s for s in sessions if s["id"].startswith(session_id)]
            if not matches:
                console.print(f"[red]No session matching '{session_id}'[/red]")
                raise typer.Exit(1)
            if len(matches) > 1:
                console.print(f"[yellow]Ambiguous prefix '{session_id}' — matches {len(matches)} sessions[/yellow]")
                raise typer.Exit(1)
            session_id = matches[0]["id"]

        session_info = await db.get_session(session_id)
        if not session_info:
            console.print(f"[red]Session not found: {session_id}[/red]")
            raise typer.Exit(1)

        mode = session_info["mode"]
        project = session_info.get("project")
        title = session_info.get("title", "")

        console.print(
            Panel(
                f"Resuming: [bold]{title or 'untitled'}[/bold]\n"
                f"Mode: {mode}" + (f" | Project: {project}" if project else ""),
                border_style="yellow",
            )
        )

        await db.close()  # REPLs manage their own DB connections

        if mode in ("agent", "chat"):
            from forge.agent.loop import agent_repl

            if mode == "chat":
                await agent_repl(resume_session_id=session_id, system=CHAT_SYSTEM)
            else:
                await agent_repl(resume_session_id=session_id)
        elif mode == "code" and project:
            # Re-enter code REPL with DB for RAG + persistence
            rag_db = Database()
            try:
                await rag_db.connect()
            except Exception:
                rag_db = None  # type: ignore[assignment]
            try:
                await _code_repl(rag_db, project, resume_session_id=session_id)
            finally:
                if rag_db:
                    await rag_db.close()
        else:
            # Legacy chat sessions or unknown modes — use tool-using REPL
            from forge.agent.loop import agent_repl

            await agent_repl(resume_session_id=session_id, system=CHAT_SYSTEM)

    except typer.Exit:
        await db.close()
        raise
    except Exception:
        await db.close()
        raise


@app.command()
def index(
    path: str = typer.Argument(".", help="Directory to index"),
    project: str = typer.Option(None, "--project", "-p", help="Project name (defaults to directory name)"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-index all files (ignore hashes)"),
) -> None:
    """Index a directory for RAG — walk, chunk, embed, store."""
    asyncio.run(_index_command(path, project, force))


async def _index_command(path: str, project: str | None, force: bool) -> None:
    from forge.rag.indexer import index_directory

    db = Database()
    try:
        await db.connect()
    except Exception as e:
        console.print(f"[red]Cannot connect to database:[/red] {e}")
        console.print("Run [bold]uv run python scripts/setup_db.py[/bold] to set up the database.")
        raise typer.Exit(1)

    try:
        stats = await index_directory(path, db, project=project, force=force)

        console.print(f"\n[bold]Indexing complete[/bold]")
        console.print(f"  Files scanned: {stats['files_scanned']}")
        console.print(f"  Files indexed: {stats['files_indexed']}")
        console.print(f"  Files skipped: {stats['files_skipped']}")
        console.print(f"  Chunks stored: {stats['chunks_stored']}")

        # Show project stats
        proj = project or __import__("pathlib").Path(path).resolve().name
        db_stats = await db.get_project_stats(proj)
        console.print(f"\n  [dim]Project '{proj}': {db_stats['chunk_count']} total chunks "
                       f"across {db_stats['file_count']} files[/dim]")
    finally:
        await db.close()


@app.command()
def draft(
    prompt: str = typer.Argument(..., help="What to build or generate"),
    project: str = typer.Option(None, "--project", "-p", help="Project for RAG context"),
    show_stages: bool = typer.Option(False, "--show-stages", "-s", help="Show draft and critique"),
) -> None:
    """Draft → Critique → Refine pipeline — two-model orchestration."""
    asyncio.run(_draft_command(prompt, project, show_stages))


async def _draft_command(prompt: str, project: str | None, show_stages: bool) -> None:
    import os

    from rich.rule import Rule
    from rich.spinner import Spinner
    from rich.status import Status

    from forge.core.pipeline import Pipeline

    fast = get_fast_backend()
    heavy = get_heavy_backend()
    pipeline = Pipeline(drafter=fast, refiner=heavy)

    # Optional RAG context
    context = ""
    db = None
    if project or True:  # Always try RAG
        proj = project or os.path.basename(os.getcwd())
        try:
            db = Database()
            await db.connect()
            chunks = await retrieve(prompt, proj, db)
            if chunks:
                context = format_context(chunks)
                console.print(f"[dim]Retrieved {len(chunks)} chunks from project '{proj}'[/dim]")
        except Exception:
            pass

    try:
        # Stage 1: Draft
        with console.status(f"[bold cyan]Drafting[/bold cyan] ({fast.name})..."):
            draft_text = await fast.generate(
                prompt,
                system=f"Generate a first draft response.\n\n{context}" if context else "Generate a first draft response.",
            )

        if show_stages:
            console.print(Rule("[bold cyan]Draft[/bold cyan]"))
            console.print(Markdown(draft_text))

        # Stage 2: Critique
        with console.status(f"[bold yellow]Critiquing[/bold yellow] ({fast.name})..."):
            from forge.prompts.refine import CRITIQUE_SYSTEM
            critique_prompt = f"## Original Request\n{prompt}\n\n## Draft Response\n{draft_text}"
            critique_text = await fast.generate(critique_prompt, system=CRITIQUE_SYSTEM)

        if show_stages:
            console.print(Rule("[bold yellow]Critique[/bold yellow]"))
            console.print(Markdown(critique_text))

        # Stage 3: Refine (streamed)
        if show_stages:
            console.print(Rule("[bold green]Refined Output[/bold green]"))
        else:
            console.print(f"[dim](draft: {fast.name} → refine: {heavy.name})[/dim]")

        from forge.prompts.refine import REFINE_SYSTEM
        refine_prompt = (
            f"## Original Request\n{prompt}\n\n"
            f"## Draft Response\n{draft_text}\n\n"
            f"## Critique\n{critique_text}"
        )
        refine_system = f"{REFINE_SYSTEM}\n\n{context}" if context else REFINE_SYSTEM

        collected: list[str] = []
        with Live(console=console, refresh_per_second=12, vertical_overflow="visible") as live:
            async for chunk in heavy.stream(refine_prompt, system=refine_system):
                collected.append(chunk)
                live.update(Markdown("".join(collected)))

    except Exception as e:
        _handle_model_error(e, "pipeline")
    finally:
        if db:
            await db.close()


@app.command()
def run(
    prompt: str = typer.Argument(..., help="What code to generate and execute"),
    retries: int = typer.Option(3, "--retries", "-r", help="Max retry attempts on failure"),
    timeout: float = typer.Option(30.0, "--timeout", "-t", help="Execution timeout in seconds"),
    project: str = typer.Option(None, "--project", "-p", help="Project for RAG context"),
) -> None:
    """Generate code, execute it, and self-correct on errors."""
    asyncio.run(_run_command(prompt, retries, timeout, project))


async def _run_command(
    prompt: str, max_retries: int, timeout: float, project: str | None
) -> None:
    import os

    from rich.rule import Rule
    from rich.syntax import Syntax

    from forge.core.executor import execute_code, extract_code, run_with_retry

    heavy = get_heavy_backend()

    # Optional RAG context
    context = ""
    db = None
    proj = project or os.path.basename(os.getcwd())
    try:
        db = Database()
        await db.connect()
        chunks = await retrieve(prompt, proj, db)
        if chunks:
            context = format_context(chunks)
            console.print(f"[dim]Retrieved {len(chunks)} chunks from project '{proj}'[/dim]")
    except Exception:
        pass

    try:
        console.print(f"[dim](executor: {heavy.name})[/dim]")

        with console.status("[bold cyan]Generating code...[/bold cyan]"):
            results = await run_with_retry(
                prompt,
                heavy,
                max_retries=max_retries,
                timeout=timeout,
                context=context,
            )

        for result in results:
            console.print(Rule(
                f"[bold]Attempt {result.attempt}[/bold] — "
                + ("[green]success[/green]" if result.success else "[red]failed[/red]")
            ))

            # Show the code
            code = result.code
            if code and not code.startswith("Could not extract"):
                console.print(Syntax(code, "python", theme="monokai", line_numbers=True))

            # Show output
            if result.stdout:
                console.print(Panel(result.stdout.rstrip(), title="stdout", border_style="green"))
            if result.stderr:
                style = "dim" if result.success else "red"
                console.print(Panel(result.stderr.rstrip(), title="stderr", border_style=style))

        # Final status
        final = results[-1] if results else None
        if final and final.success:
            console.print(f"\n[bold green]Completed successfully[/bold green] on attempt {final.attempt}")
        elif final:
            console.print(f"\n[bold red]Failed after {final.attempt} attempts[/bold red]")
        else:
            console.print("[red]No results — code generation may have failed.[/red]")

    except Exception as e:
        _handle_model_error(e, heavy.name)
    finally:
        if db:
            await db.close()
