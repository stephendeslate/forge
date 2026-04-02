"""Code REPL command with RAG context injection."""

from __future__ import annotations

import asyncio
import uuid

from rich.markdown import Markdown
from rich.panel import Panel

from forge import __version__
from forge.config import settings
from forge.core.conversation import Conversation
from forge.core.router import Route
from forge.prompts.system import CODE_SYSTEM
from forge.storage.database import Database

from ._helpers import augment_system_with_rag, console, get_router, resolve_route
from ._sessions import show_sessions
from ._streaming import generate_response, stream_response


async def code_command(
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
        force = resolve_route(gpu, fast, npu)
        system, chunk_count = await augment_system_with_rag(prompt, proj, db, CODE_SYSTEM)
        if chunk_count:
            console.print(f"[dim]Retrieved {chunk_count} chunks from project '{proj}'[/dim]")
        router = get_router()
        if settings.streaming:
            await stream_response(router, prompt, system=system, force_route=force)
        else:
            text, _ = await generate_response(router, prompt, system=system, force_route=force)
            if text:
                console.print(Markdown(text))
    else:
        await code_repl(db, proj)

    if db:
        await db.close()


async def code_repl(
    db: Database | None,
    project: str,
    resume_session_id: str | None = None,
) -> None:
    """Interactive REPL with RAG context injection."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory

    router = get_router()
    session: PromptSession[str] = PromptSession(history=InMemoryHistory())

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
                    await show_sessions(persist_db)
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

                code_system, chunk_count = await augment_system_with_rag(
                    user_input, project, db, CODE_SYSTEM,
                )
                if chunk_count:
                    console.print(f"[dim]Retrieved {chunk_count} chunks[/dim]")

                conversation.add("user", user_input)
                history = conversation.format_history()
                text, model = await stream_response(
                    router, user_input, system=code_system, force_route=force, history=history
                )
                if text:
                    conversation.add("assistant", text, model=model)
                continue
            else:
                console.print(f"[dim]Unknown command: {cmd}[/dim]")
                continue

        # Retrieve RAG context
        system, chunk_count = await augment_system_with_rag(user_input, project, db, CODE_SYSTEM)
        if chunk_count:
            console.print(f"[dim]Retrieved {chunk_count} chunks[/dim]")

        conversation.add("user", user_input)
        history = conversation.format_history()
        text, model = await stream_response(
            router, user_input, system=system, history=history
        )
        if text:
            conversation.add("assistant", text, model=model)
