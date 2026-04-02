"""Streaming and generation helpers."""

from __future__ import annotations

from rich.live import Live
from rich.markdown import Markdown

from forge.config import settings
from forge.core.router import ModelRouter, Route
from forge.prompts.system import CHAT_SYSTEM

from ._helpers import console, get_router, handle_model_error


async def stream_response(
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
        handle_model_error(e, backend.name)
        return "", backend.model_id

    full_text = "".join(collected)
    return full_text, backend.model_id


async def generate_response(
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
        handle_model_error(e, backend.name)
        return "", backend.model_id
    return result, backend.model_id


async def one_shot(
    prompt: str,
    *,
    system: str = CHAT_SYSTEM,
    force_route: Route | None = None,
    stream: bool = True,
) -> None:
    """Handle a single prompt → response."""
    from pathlib import Path

    from forge.core.project import build_project_context

    project_ctx = build_project_context(Path.cwd())
    if project_ctx:
        system = system + "\n\n" + project_ctx

    router = get_router()

    if stream and settings.streaming:
        await stream_response(router, prompt, system=system, force_route=force_route)
    else:
        text, model = await generate_response(
            router, prompt, system=system, force_route=force_route
        )
        if text:
            console.print(Markdown(text))
