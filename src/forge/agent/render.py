"""Event stream handler for rendering agent output with Rich."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterable

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
)

from forge.agent.deps import AgentDeps
from forge.agent.status import Phase

# Tool categories for color coding
_SAFE_TOOLS = {"read_file", "search_code", "list_files"}
_WRITE_TOOLS = {"write_file", "edit_file"}
_EXEC_TOOLS = {"run_command"}

# Regex to strip <think>...</think> blocks (possibly incomplete at stream end)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)


def _tool_style(tool_name: str) -> str:
    """Return a Rich border style based on tool category."""
    if tool_name in _SAFE_TOOLS:
        return "green"
    if tool_name in _WRITE_TOOLS:
        return "yellow"
    if tool_name in _EXEC_TOOLS:
        return "red"
    return "blue"


def _format_tool_args(args: str | dict | None) -> str:
    """Format tool arguments for display."""
    if args is None:
        return ""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return args

    if isinstance(args, dict):
        parts = []
        for k, v in args.items():
            val = repr(v) if isinstance(v, str) and len(str(v)) > 80 else str(v)
            if isinstance(v, str) and len(v) > 120:
                val = repr(v[:120] + "...")
            parts.append(f"  {k}: {val}")
        return "\n".join(parts)

    return str(args)


def _truncate(text: str, max_len: int = 2000) -> str:
    """Truncate text for display."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... ({len(text)} chars, truncated)"


def _split_thinking(raw: str) -> tuple[str, str]:
    """Split raw text into (thinking_content, visible_content).

    Handles partial <think> blocks (open but not yet closed).
    """
    thinking_parts: list[str] = []
    visible_parts: list[str] = []
    pos = 0
    in_think = False

    while pos < len(raw):
        if not in_think:
            m = _THINK_OPEN.search(raw, pos)
            if m:
                visible_parts.append(raw[pos : m.start()])
                pos = m.end()
                in_think = True
            else:
                visible_parts.append(raw[pos:])
                break
        else:
            m = _THINK_CLOSE.search(raw, pos)
            if m:
                thinking_parts.append(raw[pos : m.start()])
                pos = m.end()
                in_think = False
            else:
                # Still inside <think>, no closing tag yet
                thinking_parts.append(raw[pos:])
                break

    return "".join(thinking_parts).strip(), "".join(visible_parts).strip()


async def render_events(
    ctx: RunContext[AgentDeps],
    events: AsyncIterable[AgentStreamEvent],
) -> None:
    """Render agent events to the console with streaming text and tool call panels."""
    console = ctx.deps.console
    tracker = ctx.deps.status_tracker
    text_chunks: list[str] = []
    live: Live | None = None
    tool_call_count = 0
    has_thinking = False

    # Start with THINKING phase
    if tracker:
        tracker.set_phase(Phase.THINKING)

    try:
        async for event in events:
            if event.event_kind == "part_start":
                assert isinstance(event, PartStartEvent)
                if isinstance(event.part, TextPart):
                    # Start streaming text
                    if event.part.content:
                        text_chunks.append(event.part.content)

                    # Transition to STREAMING, pause tracker for Live
                    if tracker:
                        tracker.set_phase(Phase.STREAMING)
                        tracker.pause()

                    if live is None:
                        live = Live(
                            console=console,
                            refresh_per_second=12,
                            vertical_overflow="visible",
                        )
                        live.start()

                    visible = _render_text_with_thinking(
                        "".join(text_chunks), console, has_thinking
                    )
                    live.update(visible)

                elif isinstance(event.part, ToolCallPart):
                    # A tool call is starting — finalize any text stream
                    if live is not None:
                        visible = _render_text_with_thinking(
                            "".join(text_chunks), console, has_thinking
                        )
                        live.update(visible)
                        live.stop()
                        live = None
                        text_chunks.clear()
                        has_thinking = False

                    if tracker:
                        tracker.resume()
                        tracker.set_phase(Phase.TOOL_CALL, event.part.tool_name)

            elif event.event_kind == "part_delta":
                assert isinstance(event, PartDeltaEvent)
                if isinstance(event.delta, TextPartDelta):
                    text_chunks.append(event.delta.content_delta)
                    raw = "".join(text_chunks)

                    # Check if we have thinking content
                    if "<think>" in raw.lower():
                        has_thinking = True

                    if live is not None:
                        visible = _render_text_with_thinking(
                            raw, console, has_thinking
                        )
                        live.update(visible)

            elif event.event_kind == "function_tool_call":
                assert isinstance(event, FunctionToolCallEvent)
                tool_call_count += 1
                if tracker:
                    tracker.increment_tool_calls()

                # Finalize text stream if active
                if live is not None:
                    visible = _render_text_with_thinking(
                        "".join(text_chunks), console, has_thinking
                    )
                    live.update(visible)
                    live.stop()
                    live = None
                    text_chunks.clear()
                    has_thinking = False

                if tracker:
                    tracker.resume()
                    tracker.set_phase(Phase.TOOL_EXEC, event.part.tool_name)

                tool_name = event.part.tool_name
                style = _tool_style(tool_name)
                args_str = _format_tool_args(event.part.args)

                console.print(
                    Panel(
                        Text(args_str, style="dim") if args_str else Text("(no args)", style="dim"),
                        title=f"[bold]{tool_name}[/bold]",
                        border_style=style,
                        padding=(0, 1),
                    )
                )

            elif event.event_kind == "function_tool_result":
                assert isinstance(event, FunctionToolResultEvent)
                content = event.result.content if hasattr(event.result, "content") else str(event.result)
                content_str = str(content) if content else "(no output)"
                content_str = _truncate(content_str)
                outcome = getattr(event.result, "outcome", "success")
                result_style = "dim green" if outcome == "success" else "dim red"

                console.print(
                    Panel(
                        Text(content_str, style="dim"),
                        title="[dim]result[/dim]",
                        border_style=result_style,
                        padding=(0, 1),
                    )
                )

                # Back to thinking after tool result
                if tracker:
                    tracker.set_phase(Phase.THINKING)

            elif event.event_kind == "final_result":
                # Finalize any remaining text
                if live is not None:
                    visible = _render_text_with_thinking(
                        "".join(text_chunks), console, has_thinking
                    )
                    live.update(visible)
                    live.stop()
                    live = None
                    text_chunks.clear()

    finally:
        if live is not None:
            if text_chunks:
                visible = _render_text_with_thinking(
                    "".join(text_chunks), console, has_thinking
                )
                live.update(visible)
            live.stop()

        # Resume tracker so caller can stop it cleanly
        if tracker:
            tracker.resume()

    # Summary line (tracker replaces old tool count)
    if tracker:
        tracker.set_phase(Phase.DONE)
    elif tool_call_count > 0:
        console.print(f"[dim]({tool_call_count} tool call{'s' if tool_call_count != 1 else ''})[/dim]")


def _render_text_with_thinking(
    raw: str, console: Console, has_thinking: bool
) -> Markdown | Text:
    """Render streamed text, handling <think> blocks."""
    if not has_thinking:
        return Markdown(raw)

    thinking, visible = _split_thinking(raw)

    # Build a combined renderable
    from rich.console import Group

    parts = []
    if thinking:
        parts.append(
            Panel(
                Text(thinking, style="dim italic"),
                title="[dim]thinking[/dim]",
                border_style="dim",
                padding=(0, 1),
            )
        )
    if visible:
        parts.append(Markdown(visible))

    if not parts:
        return Markdown("")

    return Group(*parts)  # type: ignore[return-value]
