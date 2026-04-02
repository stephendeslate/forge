"""Shared helpers for CLI submodules."""

from __future__ import annotations

from rich.console import Console

from forge.config import settings
from forge.core.router import ModelRouter, Route
from forge.models.npu import get_npu_backend
from forge.models.ollama import get_fast_backend, get_heavy_backend
from forge.storage.database import Database

console = Console()


def get_router() -> ModelRouter:
    return ModelRouter(heavy=get_heavy_backend(), fast=get_fast_backend(), npu=get_npu_backend())


def resolve_route(gpu: bool, fast: bool, npu: bool = False) -> Route | None:
    if npu:
        return Route.NPU
    if gpu:
        return Route.HEAVY
    if fast:
        return Route.FAST
    return None


def handle_model_error(e: Exception, backend_name: str) -> None:
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


async def augment_system_with_rag(
    prompt: str, project: str, db: Database | None, system: str,
) -> tuple[str, int]:
    """Retrieve RAG chunks and prepend to system prompt. Returns (system, chunk_count)."""
    if not db:
        return system, 0
    try:
        from forge.rag.retriever import format_context, retrieve

        chunks = await retrieve(prompt, project, db)
        if chunks:
            ctx = format_context(chunks)
            return f"{system}\n\n{ctx}", len(chunks)
    except Exception:
        pass
    return system, 0
