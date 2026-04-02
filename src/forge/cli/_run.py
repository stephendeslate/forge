"""Run command — generate, execute, and self-correct code."""

from __future__ import annotations

from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax

from forge.models.ollama import get_heavy_backend
from forge.storage.database import Database

from ._helpers import console, handle_model_error


async def run_command(
    prompt: str, max_retries: int, timeout: float, project: str | None
) -> None:
    import os

    from forge.core.executor import execute_code, extract_code, run_with_retry

    heavy = get_heavy_backend()

    # Optional RAG context
    context = ""
    db = None
    proj = project or os.path.basename(os.getcwd())
    try:
        db = Database()
        await db.connect()
        from forge.rag.retriever import format_context, retrieve

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

            code = result.code
            if code and not code.startswith("Could not extract"):
                console.print(Syntax(code, "python", theme="monokai", line_numbers=True))

            if result.stdout:
                console.print(Panel(result.stdout.rstrip(), title="stdout", border_style="green"))
            if result.stderr:
                style = "dim" if result.success else "red"
                console.print(Panel(result.stderr.rstrip(), title="stderr", border_style=style))

        final = results[-1] if results else None
        if final and final.success:
            console.print(f"\n[bold green]Completed successfully[/bold green] on attempt {final.attempt}")
        elif final:
            console.print(f"\n[bold red]Failed after {final.attempt} attempts[/bold red]")
        else:
            console.print("[red]No results — code generation may have failed.[/red]")

    except Exception as e:
        handle_model_error(e, heavy.name)
    finally:
        if db:
            await db.close()
