"""Index command — walk, chunk, embed, store."""

from __future__ import annotations

import typer

from forge.storage.database import Database

from ._helpers import console


async def index_command(path: str, project: str | None, force: bool) -> None:
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

        proj = project or __import__("pathlib").Path(path).resolve().name
        db_stats = await db.get_project_stats(proj)
        console.print(f"\n  [dim]Project '{proj}': {db_stats['chunk_count']} total chunks "
                       f"across {db_stats['file_count']} files[/dim]")
    finally:
        await db.close()
