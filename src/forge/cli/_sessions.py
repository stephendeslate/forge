"""Session management — history, resume, session display."""

from __future__ import annotations

import typer
from rich.panel import Panel
from rich.table import Table

from forge.prompts.system import CHAT_SYSTEM
from forge.storage.database import Database

from ._helpers import console


def format_age(dt: object) -> str:
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


async def show_sessions(db: Database, limit: int = 20) -> None:
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
        turns = s["message_count"] // 2
        table.add_row(
            s["id"][:8],
            s["title"] or "[dim]untitled[/dim]",
            s["mode"],
            str(turns),
            format_age(s["updated_at"]),
        )

    console.print(table)


async def history_command(limit: int) -> None:
    db = Database()
    try:
        await db.connect()
    except Exception as e:
        console.print(f"[red]Cannot connect to database:[/red] {e}")
        raise typer.Exit(1)

    try:
        await show_sessions(db, limit=limit)
    finally:
        await db.close()


async def resume_command(session_id: str | None) -> None:
    db = Database()
    try:
        await db.connect()
    except Exception as e:
        console.print(f"[red]Cannot connect to database:[/red] {e}")
        raise typer.Exit(1)

    try:
        if not session_id:
            sid = await db.get_latest_session_id()
            if not sid:
                console.print("[dim]No sessions found.[/dim]")
                raise typer.Exit(0)
            session_id = sid
        else:
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
            from ._code import code_repl

            rag_db = Database()
            try:
                await rag_db.connect()
            except Exception:
                rag_db = None  # type: ignore[assignment]
            try:
                await code_repl(rag_db, project, resume_session_id=session_id)
            finally:
                if rag_db:
                    await rag_db.close()
        else:
            from forge.agent.loop import agent_repl

            await agent_repl(resume_session_id=session_id, system=CHAT_SYSTEM)

    except typer.Exit:
        await db.close()
        raise
    except Exception:
        await db.close()
        raise
