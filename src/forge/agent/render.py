"""Event stream handler for rendering agent output with Rich."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterable

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
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

# Read-only tools get compact result display
_READ_TOOLS = {"read_file", "search_code", "list_files"}

# Diff-producing tools get syntax-highlighted results
_DIFF_TOOLS = {"edit_file"}

# Icons for tool categories
_TOOL_ICONS: dict[str, str] = {
    "read_file": "📖",
    "search_code": "🔍",
    "list_files": "📂",
    "write_file": "📝",
    "edit_file": "✏️ ",
    "run_command": "▶",
    "web_search": "🌐",
    "web_fetch": "🌐",
    "rag_search": "🔍",
    "save_memory": "💾",
    "recall_memories": "💭",
    "task_create": "📋",
    "task_update": "📋",
    "task_list": "📋",
    "task_get": "📋",
    "delegate": "🔀",
    "delegate_parallel": "🔀",
}


def _tool_icon(tool_name: str) -> str:
    """Return an icon for the tool, falling back to category-based icons."""
    if tool_name in _TOOL_ICONS:
        return _TOOL_ICONS[tool_name]
    # MCP tools (prefixed with server name)
    if "_" in tool_name:
        return "🔌"
    return "🔧"

# Regex to strip <think>...</think> blocks (possibly incomplete at stream end)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)


def _find_safe_boundary(text: str) -> int:
    """Find a safe markdown boundary to render up to, avoiding mid-block splits.

    Returns the index up to which it's safe to render, or 0 to wait for more data.
    """
    if text.endswith("\n"):
        return len(text)

    # Prefer paragraph boundary
    pp = text.rfind("\n\n")
    if pp >= 0:
        return pp + 2

    # Fall back to line boundary
    nl = text.rfind("\n")
    if nl >= 0:
        return nl + 1

    # If no newlines but very long, flush anyway to prevent stalls
    if len(text) > 200:
        return len(text)

    return 0


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


def _truncate(text: str, max_len: int = 500) -> str:
    """Truncate text for display using head+tail strategy."""
    from forge.agent.utils import head_tail_truncate

    return head_tail_truncate(text, max_len)


def _extract_diff_block(text: str) -> str | None:
    """Extract unified diff content from markdown fenced diff blocks."""
    # Match ```diff\n...\n``` blocks produced by edit_file
    m = re.search(r"```diff\n(.*?)\n```", text, re.DOTALL)
    return m.group(1) if m else None


def _format_result_renderable(
    content_str: str, tool_name: str, result_style: str,
    duration_badge: str = "",
) -> Panel:
    """Build a result panel with tool-aware formatting.

    - edit_file results: extract diff and render with syntax highlighting
    - read-only tool results: compact display with dimmer border
    """
    from rich.console import Group

    outcome_icon = "✓" if "green" in result_style else "✗"
    badge = f" {duration_badge}" if duration_badge else ""

    # For diff-producing tools, render the diff with syntax highlighting
    if tool_name in _DIFF_TOOLS:
        diff_text = _extract_diff_block(content_str)
        if diff_text:
            parts = []
            # Show any text before the diff block
            before = content_str.split("```diff\n", 1)[0].strip()
            if before:
                parts.append(Text(before, style="dim"))
            parts.append(
                Syntax(diff_text, "diff", theme="monokai", line_numbers=False)
            )
            return Panel(
                Group(*parts) if len(parts) > 1 else parts[0],
                title=f"[dim]{outcome_icon} result{badge}[/dim]",
                border_style=result_style,
                padding=(0, 1),
            )

    # For read-only tools, use a dimmer compact style
    if tool_name in _READ_TOOLS:
        return Panel(
            Text(content_str, style="dim"),
            title=f"[dim]{outcome_icon} {tool_name}{badge}[/dim]",
            border_style="dim green",
            padding=(0, 1),
        )

    # Default: plain text result
    return Panel(
        Text(content_str, style="dim"),
        title=f"[dim]{outcome_icon} result{badge}[/dim]",
        border_style=result_style,
        padding=(0, 1),
    )


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
    import time

    console = ctx.deps.console
    tracker = ctx.deps.status_tracker
    text_chunks: list[str] = []
    live: Live | None = None
    thinking_live: Live | None = None
    tool_call_count = 0
    has_thinking = False
    last_tool_name: str = ""
    tool_start_time: float = 0.0

    def _stop_thinking_spinner() -> None:
        nonlocal thinking_live
        if thinking_live is not None:
            thinking_live.stop()
            thinking_live = None
            if tracker:
                tracker.resume()

    def _finalize_live() -> None:
        """Stop non-transient Live and record streamed text in the turn buffer.

        The Live persists on screen (transient=False). We add the final
        rendered content to the turn buffer as already_printed so that
        loop.py won't duplicate it and Ctrl-R rerender still works.
        """
        nonlocal live, has_thinking
        if live is None:
            return
        raw = "".join(text_chunks)
        if raw.strip():
            visible = _render_text_with_thinking(raw, console, has_thinking)
            live.update(visible)
            turn_buffer = ctx.deps.turn_buffer
            if turn_buffer:
                turn_buffer.add(visible, is_tool=False, already_printed=True)
        live.stop()
        live = None
        text_chunks.clear()
        has_thinking = False

    # Start with THINKING phase
    if tracker:
        tracker.set_phase(Phase.THINKING)

    try:
        async for event in events:
            if event.event_kind == "part_start":
                assert isinstance(event, PartStartEvent)
                if isinstance(event.part, TextPart):
                    _stop_thinking_spinner()
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
                            transient=False,
                        )
                        live.start()

                    visible = _render_text_with_thinking(
                        "".join(text_chunks), console, has_thinking
                    )
                    live.update(visible)

                elif isinstance(event.part, ToolCallPart):
                    _stop_thinking_spinner()
                    _finalize_live()

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
                        # Use safe boundary to avoid mid-markdown rendering artifacts
                        if has_thinking:
                            visible = _render_text_with_thinking(
                                raw, console, has_thinking
                            )
                        else:
                            safe_end = _find_safe_boundary(raw)
                            if safe_end > 0:
                                visible = Markdown(raw[:safe_end])
                            else:
                                visible = Markdown(raw)
                        live.update(visible)

            elif event.event_kind == "function_tool_call":
                _stop_thinking_spinner()
                assert isinstance(event, FunctionToolCallEvent)
                tool_call_count += 1
                if tracker:
                    tracker.increment_tool_calls()

                _finalize_live()

                tool_name = event.part.tool_name
                last_tool_name = tool_name
                tool_start_time = time.monotonic()
                style = _tool_style(tool_name)
                icon = _tool_icon(tool_name)
                args_str = _format_tool_args(event.part.args)

                # Build the full panel with icon in title
                title = f"{icon} [bold]{tool_name}[/bold]"
                if args_str:
                    panel = Panel(
                        Text(args_str, style="dim"),
                        title=title,
                        border_style=style,
                        padding=(0, 1),
                    )
                else:
                    panel = Panel(
                        Text("(no args)", style="dim"),
                        title=title,
                        border_style=style,
                        padding=(0, 1),
                    )

                printed = False
                if ctx.deps.tools_visible:
                    if tracker:
                        tracker.pause()
                    console.print(panel)
                    printed = True
                    if tracker:
                        tracker.resume()

                # Store in turn buffer for rerender
                turn_buffer = ctx.deps.turn_buffer
                if turn_buffer:
                    turn_buffer.add(panel, is_tool=True, already_printed=printed)

                if tracker:
                    tracker.set_phase(Phase.TOOL_EXEC, tool_name)

            elif event.event_kind == "function_tool_result":
                assert isinstance(event, FunctionToolResultEvent)
                content = event.result.content if hasattr(event.result, "content") else str(event.result)
                content_str = str(content) if content else "(no output)"
                outcome = getattr(event.result, "outcome", "success")
                result_style = "dim green" if outcome == "success" else "dim red"

                # Calculate duration since tool call started
                duration = time.monotonic() - tool_start_time if tool_start_time > 0 else 0.0
                duration_badge = f" [dim]({duration:.1f}s)[/dim]" if duration >= 0.1 else ""

                # Get tool name from the result part or fall back to tracked name
                result_tool_name = getattr(event.result, "tool_name", last_tool_name)

                # For diff tools, use higher truncation limit to preserve diff fences
                trunc_limit = 2000 if result_tool_name in _DIFF_TOOLS else 500
                content_str = _truncate(content_str, trunc_limit)

                result_panel = _format_result_renderable(
                    content_str, result_tool_name, result_style,
                    duration_badge=duration_badge,
                )

                printed = False
                if ctx.deps.tools_visible:
                    if tracker:
                        tracker.pause()
                    console.print(result_panel)
                    printed = True
                    if tracker:
                        tracker.resume()

                # Store in turn buffer for rerender
                turn_buffer = ctx.deps.turn_buffer
                if turn_buffer:
                    turn_buffer.add(result_panel, is_tool=True, already_printed=printed)

                if tracker:
                    tracker.set_phase(Phase.THINKING)

                # Show inline thinking spinner only when status line is hidden
                if not ctx.deps.status_visible:
                    if tracker:
                        tracker.pause()
                    thinking_live = Live(
                        Spinner("dots", text="thinking...", style="dim"),
                        console=console,
                        refresh_per_second=10,
                        transient=True,
                    )
                    thinking_live.start()

            elif event.event_kind == "final_result":
                _stop_thinking_spinner()
                # Do NOT call _finalize_live() here — pydantic-ai may emit
                # text deltas AFTER final_result. The finally block handles
                # finalization after all events have been consumed.

    finally:
        _stop_thinking_spinner()
        _finalize_live()

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
