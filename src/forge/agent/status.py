"""Real-time status line for agent execution."""

from __future__ import annotations

import asyncio
import fcntl
import os
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console


class Phase(Enum):
    """Current phase of agent execution."""

    THINKING = "thinking"
    STREAMING = "streaming"
    TOOL_CALL = "tool call"
    TOOL_EXEC = "executing"
    DONE = "done"


# Phase display config: (emoji, style)
_PHASE_DISPLAY: dict[Phase, tuple[str, str]] = {
    Phase.THINKING: ("🧠", "dim"),
    Phase.STREAMING: ("✍️ ", "dim"),
    Phase.TOOL_CALL: ("🔧", "dim"),
    Phase.TOOL_EXEC: ("⏳", "dim"),
    Phase.DONE: ("✅", "dim"),
}


def _set_nonblocking(fd: int) -> int:
    """Set fd to non-blocking mode, return old flags."""
    old = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, old | os.O_NONBLOCK)
    return old


def _restore_flags(fd: int, flags: int) -> None:
    """Restore original fd flags."""
    fcntl.fcntl(fd, fcntl.F_SETFL, flags)


def _try_read_byte(fd: int) -> int | None:
    """Try to read one byte from fd (non-blocking). Returns byte value or None."""
    try:
        data = os.read(fd, 1)
        return data[0] if data else None
    except (BlockingIOError, OSError):
        return None


@dataclass
class StatusTracker:
    """Ephemeral ANSI status line showing phase + elapsed time + tool count.

    Prints an overwriting status line via \\r\\033[2K. Pauses when Rich Live
    is active (streaming text) and resumes between events.
    """

    console: Console
    visible: bool = True
    _start_time: float = field(default=0.0, init=False)
    _phase: Phase = field(default=Phase.THINKING, init=False)
    _detail: str = field(default="", init=False)
    _tool_calls: int = field(default=0, init=False)
    _ticker: asyncio.Task[None] | None = field(default=None, init=False)
    _key_monitor: asyncio.Task[None] | None = field(default=None, init=False)
    _paused: bool = field(default=False, init=False)
    _active: bool = field(default=False, init=False)
    _on_toggle: object | None = field(default=None, init=False)  # callback
    _on_tools_toggle: object | None = field(default=None, init=False)  # callback

    @property
    def tool_calls(self) -> int:
        return self._tool_calls

    def start(
        self,
        on_toggle: object | None = None,
        on_tools_toggle: object | None = None,
    ) -> None:
        """Start the status ticker and optional keypress monitor.

        Args:
            on_toggle: Callable invoked with (visible: bool) when Ctrl-O is pressed.
            on_tools_toggle: Callable invoked (no args) when Ctrl-R is pressed.
        """
        self._start_time = time.monotonic()
        self._active = True
        self._on_toggle = on_toggle
        self._on_tools_toggle = on_tools_toggle
        if not self.console.is_terminal:
            return
        self._phase = Phase.THINKING
        self._paused = False
        try:
            loop = asyncio.get_running_loop()
            self._ticker = loop.create_task(self._tick_loop())
            self._key_monitor = loop.create_task(self._key_loop())
        except RuntimeError:
            # No running event loop — skip ticker (non-async context)
            pass

    def stop(self) -> None:
        """Stop the ticker, key monitor, and clear the status line."""
        self._active = False
        for task in (self._ticker, self._key_monitor):
            if task and not task.done():
                task.cancel()
        self._ticker = None
        self._key_monitor = None
        self._clear_line()

    def set_phase(self, phase: Phase, detail: str = "") -> None:
        """Update current phase and immediately print."""
        self._phase = phase
        self._detail = detail
        if self._active and not self._paused:
            self._print_status()

    def pause(self) -> None:
        """Pause status line output (for Rich Live compatibility)."""
        if self._active:
            self._paused = True
            self._clear_line()

    def resume(self) -> None:
        """Resume status line output after Rich Live stops."""
        if self._active:
            self._paused = False
            if self.visible:
                self._print_status()

    def increment_tool_calls(self) -> None:
        """Increment the tool call counter."""
        self._tool_calls += 1

    def summary(self) -> str:
        """Return a final summary string."""
        elapsed = time.monotonic() - self._start_time
        tc = self._tool_calls
        if tc == 0:
            return f"[dim]{elapsed:.1f}s[/dim]"
        calls = "call" if tc == 1 else "calls"
        return f"[dim]{tc} tool {calls} in {elapsed:.1f}s[/dim]"

    def _elapsed(self) -> str:
        return f"{time.monotonic() - self._start_time:.1f}s"

    def _print_status(self) -> None:
        """Print the ephemeral status line."""
        if not self.console.is_terminal or self._paused or not self._active or not self.visible:
            return
        emoji, _ = _PHASE_DISPLAY.get(self._phase, ("", "dim"))
        parts = [emoji, self._elapsed(), "│", self._phase.value]
        if self._detail:
            parts.extend(["│", self._detail])
        tc = self._tool_calls
        if tc > 0:
            calls = "call" if tc == 1 else "calls"
            parts.extend(["│", f"{tc} tool {calls}"])
        line = " ".join(parts)
        # Write directly to stderr to avoid Rich markup processing
        sys.stderr.write(f"\r\033[2K\033[2m{line}\033[0m")
        sys.stderr.flush()

    def _clear_line(self) -> None:
        """Clear the status line."""
        if self.console.is_terminal:
            sys.stderr.write("\r\033[2K")
            sys.stderr.flush()

    async def _tick_loop(self) -> None:
        """Tick the status line every 100ms."""
        try:
            while self._active:
                if not self._paused:
                    self._print_status()
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    async def _key_loop(self) -> None:
        """Monitor stdin for Ctrl-O (0x0f) to toggle visibility."""
        if not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            # Set cbreak mode — passes keys immediately, no echo
            tty.setcbreak(fd)
            # Make stdin non-blocking so we can poll without freezing
            old_flags = _set_nonblocking(fd)
            try:
                while self._active:
                    ch = _try_read_byte(fd)
                    if ch == 0x0F:  # Ctrl-O
                        self.visible = not self.visible
                        if not self.visible:
                            self._clear_line()
                        if self._on_toggle:
                            self._on_toggle(self.visible)  # type: ignore[operator]
                    elif ch == 0x12:  # Ctrl-R
                        if self._on_tools_toggle:
                            self._on_tools_toggle()  # type: ignore[operator]
                    await asyncio.sleep(0.05)
            finally:
                _restore_flags(fd, old_flags)
        except asyncio.CancelledError:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
