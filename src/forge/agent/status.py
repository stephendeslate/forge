"""Real-time status bar for agent execution."""

from __future__ import annotations

import asyncio
import os
import sys
import time

try:
    import fcntl
    import termios
    import tty
    _HAS_TERMIOS = True
except ImportError:
    _HAS_TERMIOS = False
from collections.abc import Callable
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


# Phase display config: (emoji, ansi_color_code)
_PHASE_DISPLAY: dict[Phase, tuple[str, str]] = {
    Phase.THINKING: ("◆", "\033[35m"),      # magenta
    Phase.STREAMING: ("▸", "\033[36m"),      # cyan
    Phase.TOOL_CALL: ("⚙", "\033[33m"),     # yellow
    Phase.TOOL_EXEC: ("⏳", "\033[33m"),     # yellow
    Phase.DONE: ("✓", "\033[32m"),           # green
}

# Spinner frames for animated phase indicator
_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


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
    """Persistent bottom status bar showing phase, model, tokens, and elapsed time.

    Renders a formatted bar via \\r\\033[2K using ANSI escape codes. Pauses when
    Rich Live is active (streaming text) and resumes between events.
    """

    console: Console
    visible: bool = True
    model_name: str = ""
    token_budget: int = 0
    mode: str = ""
    _start_time: float = field(default=0.0, init=False)
    _phase: Phase = field(default=Phase.THINKING, init=False)
    _detail: str = field(default="", init=False)
    _tool_calls: int = field(default=0, init=False)
    _ticker: asyncio.Task[None] | None = field(default=None, init=False)
    _key_monitor: asyncio.Task[None] | None = field(default=None, init=False)
    _paused: bool = field(default=False, init=False)
    _active: bool = field(default=False, init=False)
    _on_toggle: Callable[[bool], None] | None = field(default=None, init=False)
    _on_tools_toggle: Callable[[], None] | None = field(default=None, init=False)
    _spinner_idx: int = field(default=0, init=False)
    # Token tracking — set externally after each turn
    tokens_in: int = field(default=0, init=False)
    tokens_out: int = field(default=0, init=False)

    @property
    def tool_calls(self) -> int:
        return self._tool_calls

    def start(
        self,
        on_toggle: Callable[[bool], None] | None = None,
        on_tools_toggle: Callable[[], None] | None = None,
    ) -> None:
        """Start the status ticker and optional keypress monitor."""
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
        """Return a final summary string with Rich markup."""
        elapsed = time.monotonic() - self._start_time
        tc = self._tool_calls
        parts: list[str] = []
        if tc > 0:
            calls = "call" if tc == 1 else "calls"
            parts.append(f"[dim cyan]{tc} tool {calls}[/dim cyan]")
        if self.tokens_in > 0 or self.tokens_out > 0:
            parts.append(f"[dim]{self.tokens_in:,}↑ {self.tokens_out:,}↓ tok[/dim]")
        parts.append(f"[dim]{elapsed:.1f}s[/dim]")
        return " · ".join(parts)

    def _elapsed(self) -> str:
        return f"{time.monotonic() - self._start_time:.1f}s"

    def _context_bar(self) -> str:
        """Build a mini context usage bar like [████░░░░ 45%]."""
        if self.token_budget <= 0 or self.tokens_in <= 0:
            return ""
        pct = min(self.tokens_in / self.token_budget, 1.0)
        bar_width = 8
        filled = int(pct * bar_width)
        empty = bar_width - filled

        # Color gradient: green < 60%, yellow 60-80%, red > 80%
        if pct < 0.6:
            color = "\033[32m"  # green
        elif pct < 0.8:
            color = "\033[33m"  # yellow
        else:
            color = "\033[31m"  # red

        bar = f"{color}{'█' * filled}{'░' * empty}\033[0m"
        return f" [{bar} \033[2m{pct:.0%}\033[0m]"

    def _print_status(self) -> None:
        """Print the formatted status bar."""
        if not self.console.is_terminal or self._paused or not self._active or not self.visible:
            return

        icon, color = _PHASE_DISPLAY.get(self._phase, ("·", "\033[2m"))

        # Animated spinner for active phases
        if self._phase in (Phase.THINKING, Phase.TOOL_EXEC):
            spinner = _SPINNER_FRAMES[self._spinner_idx % len(_SPINNER_FRAMES)]
            self._spinner_idx += 1
        else:
            spinner = icon

        # Build segments
        reset = "\033[0m"
        dim = "\033[2m"

        # Phase + detail
        phase_str = f"{color}{spinner}{reset} {dim}{self._phase.value}{reset}"
        if self._detail:
            phase_str += f" {dim}│{reset} {color}{self._detail}{reset}"

        # Elapsed
        elapsed_str = f"{dim}{self._elapsed()}{reset}"

        # Tool count
        tc_str = ""
        if self._tool_calls > 0:
            calls = "call" if self._tool_calls == 1 else "calls"
            tc_str = f" {dim}│{reset} {dim}{self._tool_calls} tool {calls}{reset}"

        # Token counts
        tok_str = ""
        if self.tokens_in > 0 or self.tokens_out > 0:
            tok_str = f" {dim}│ {self.tokens_in:,}↑ {self.tokens_out:,}↓{reset}"

        # Context bar
        ctx_bar = self._context_bar()

        # Model badge
        model_str = ""
        if self.model_name:
            # Shorten model name for display
            short = self.model_name
            if ":" in short:
                parts = short.split(":")
                short = parts[-1] if parts[-1] else parts[0]
            if len(short) > 20:
                short = short[:18] + "…"
            model_str = f" {dim}│{reset} \033[36m{short}{reset}"

        # Mode badge
        mode_str = ""
        if self.mode:
            mode_labels = {"local": "LOCAL", "balanced": "HYBRID", "max": "MAX"}
            mode_colors = {"local": "\033[32m", "balanced": "\033[36m", "max": "\033[35m"}
            label = mode_labels.get(self.mode, self.mode.upper())
            mcolor = mode_colors.get(self.mode, dim)
            mode_str = f" {mcolor}[{label}]{reset}"

        line = f" {phase_str} {dim}│{reset} {elapsed_str}{tc_str}{tok_str}{ctx_bar}{model_str}{mode_str} "

        sys.stderr.write(f"\r\033[2K{line}")
        sys.stderr.flush()

    def _clear_line(self) -> None:
        """Clear the status line."""
        if self.console.is_terminal:
            sys.stderr.write("\r\033[2K")
            sys.stderr.flush()

    async def _tick_loop(self) -> None:
        """Tick the status line every 80ms for smooth spinner animation."""
        try:
            while self._active:
                if not self._paused:
                    self._print_status()
                await asyncio.sleep(0.08)
        except asyncio.CancelledError:
            pass

    async def _key_loop(self) -> None:
        """Monitor stdin for Ctrl-O (0x0f) to toggle visibility."""
        if not _HAS_TERMIOS or not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        old_settings = None
        try:
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except (termios.error, OSError):
            return
        try:
            old_flags = _set_nonblocking(fd)
            try:
                while self._active:
                    ch = _try_read_byte(fd)
                    if ch == 0x0F:  # Ctrl-O
                        self.visible = not self.visible
                        if not self.visible:
                            self._clear_line()
                        if self._on_toggle:
                            self._on_toggle(self.visible)
                    elif ch == 0x12:  # Ctrl-R
                        if self._on_tools_toggle:
                            self._on_tools_toggle()
                    await asyncio.sleep(0.05)
            finally:
                _restore_flags(fd, old_flags)
        except asyncio.CancelledError:
            pass
        finally:
            if old_settings is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
