"""Draft → Critique → Refine pipeline command."""

from __future__ import annotations

from rich.live import Live
from rich.markdown import Markdown
from rich.rule import Rule

from forge.models.ollama import get_fast_backend, get_heavy_backend
from forge.storage.database import Database

from ._helpers import augment_system_with_rag, console, handle_model_error


async def draft_command(prompt: str, project: str | None, show_stages: bool) -> None:
    import os

    from forge.core.pipeline import Pipeline

    fast = get_fast_backend()
    heavy = get_heavy_backend()
    pipeline = Pipeline(drafter=fast, refiner=heavy)

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
        handle_model_error(e, "pipeline")
    finally:
        if db:
            await db.close()
