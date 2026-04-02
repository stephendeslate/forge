"""TurnBuffer — stores renderables from a turn and can erase + reprint."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group


@dataclass
class _Item:
    renderable: Any
    is_tool: bool
    was_printed: bool = False


@dataclass
class TurnBuffer:
    """Accumulates renderables for a single agent turn.

    Tracks which items are tool-related so the entire turn can be
    re-rendered with tool panels shown or hidden (Ctrl+R toggle).

    Items can be added as already-printed (live during execution) or
    not-yet-printed (answer/summary added after execution).
    """

    console: Console
    _items: list[_Item] = field(default_factory=list)
    _printed_lines: int = 0  # total lines currently on screen from this turn

    def add(
        self, renderable: Any, *, is_tool: bool = False, already_printed: bool = False
    ) -> None:
        """Store a renderable tagged as tool or non-tool content.

        If already_printed=True, the item is assumed to already be on screen
        and its line count is added to _printed_lines for erase tracking.
        """
        item = _Item(renderable=renderable, is_tool=is_tool, was_printed=already_printed)
        self._items.append(item)
        if already_printed:
            self._printed_lines += self._count_lines(renderable)

    def print_final(self, tools_visible: bool) -> None:
        """Print items that haven't been printed yet, respecting tools_visible filter."""
        to_print = [
            it.renderable
            for it in self._items
            if not it.was_printed and (not it.is_tool or tools_visible)
        ]
        if not to_print:
            return

        group = Group(*to_print)
        new_lines = self._count_lines(group)
        self._printed_lines += new_lines
        self.console.print(group)

        # Mark as printed
        for it in self._items:
            if not it.was_printed:
                it.was_printed = True

    def rerender(self, tools_visible: bool, extra_lines: int = 0) -> None:
        """Erase all printed output and reprint everything with new visibility.

        Args:
            extra_lines: Additional lines to erase above the buffer content
                         (e.g., prompt lines from prompt_toolkit).
        """
        total_erase = self._printed_lines + extra_lines
        if total_erase > 0:
            out = self.console.file
            out.write(f"\033[{total_erase}A\033[J")  # type: ignore[union-attr]
            out.flush()  # type: ignore[union-attr]

        # Reset and reprint everything
        self._printed_lines = 0
        for it in self._items:
            it.was_printed = False

        filtered = [
            it.renderable for it in self._items if not it.is_tool or tools_visible
        ]
        if not filtered:
            return

        group = Group(*filtered)
        self._printed_lines = self._count_lines(group)
        self.console.print(group)

        for it in self._items:
            it.was_printed = True

    def clear(self) -> None:
        """Reset the buffer for a new turn."""
        self._items.clear()
        self._printed_lines = 0

    def _count_lines(self, renderable: Any) -> int:
        """Count how many terminal lines a renderable will occupy."""
        buf = io.StringIO()
        measure = Console(
            file=buf,
            width=self.console.width,
            force_terminal=True,
            no_color=True,
        )
        measure.print(renderable)
        return buf.getvalue().count("\n")
