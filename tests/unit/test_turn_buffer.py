"""Tests for TurnBuffer — renderable storage and rerender."""

from __future__ import annotations

import io

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from forge.agent.turn_buffer import TurnBuffer


def _make_buffer(width: int = 80) -> tuple[TurnBuffer, io.StringIO]:
    """Create a TurnBuffer backed by a StringIO console."""
    out = io.StringIO()
    console = Console(file=out, width=width, force_terminal=True, no_color=True)
    return TurnBuffer(console=console), out


def test_add_stores_items():
    buf, _ = _make_buffer()
    buf.add(Text("hello"), is_tool=False)
    buf.add(Panel("tool output"), is_tool=True)
    assert len(buf._items) == 2
    assert buf._items[0].is_tool is False
    assert buf._items[1].is_tool is True


def test_print_final_only_prints_unprinted():
    """Items marked already_printed should not be re-printed by print_final."""
    buf, out = _make_buffer()
    buf.add(Panel("tool call"), is_tool=True, already_printed=True)
    buf.add(Text("answer text"), is_tool=False)
    buf.print_final(tools_visible=True)
    output = out.getvalue()
    # The tool panel was already_printed, so print_final should only print the answer
    assert "answer text" in output
    assert "tool call" not in output


def test_print_final_filters_hidden_tools():
    buf, out = _make_buffer()
    buf.add(Text("answer text"), is_tool=False)
    buf.add(Panel("tool stuff", title="read_file"), is_tool=True)
    buf.print_final(tools_visible=False)
    output = out.getvalue()
    assert "answer text" in output
    assert "tool stuff" not in output


def test_print_final_shows_all_when_visible():
    buf, out = _make_buffer()
    buf.add(Text("answer text"), is_tool=False)
    buf.add(Panel("tool stuff", title="read_file"), is_tool=True)
    buf.print_final(tools_visible=True)
    output = out.getvalue()
    assert "answer text" in output
    assert "tool stuff" in output


def test_print_final_tracks_lines():
    buf, _ = _make_buffer()
    buf.add(Text("line one"), is_tool=False)
    buf.add(Text("line two"), is_tool=False)
    buf.print_final(tools_visible=True)
    assert buf._printed_lines > 0


def test_already_printed_counts_lines():
    """Items marked already_printed should contribute to _printed_lines."""
    buf, _ = _make_buffer()
    buf.add(Panel("tool output"), is_tool=True, already_printed=True)
    assert buf._printed_lines > 0


def test_rerender_erases_and_reprints():
    buf, out = _make_buffer()
    # Simulate live-printed tool panel
    buf.add(Panel("tool result"), is_tool=True, already_printed=True)
    # Simulate end-of-turn answer
    buf.add(Text("answer"), is_tool=False)
    buf.print_final(tools_visible=True)
    first_output = out.getvalue()
    assert "answer" in first_output

    # Rerender with tools hidden — should contain ANSI erase sequence
    buf.rerender(tools_visible=False)
    full_output = out.getvalue()
    assert "\033[J" in full_output  # Clear-below escape
    # The new output after erase should not contain tool result
    after_erase = full_output[len(first_output):]
    assert "tool result" not in after_erase
    assert "answer" in after_erase


def test_rerender_toggle_back():
    """Rerender with tools visible again should show everything."""
    buf, out = _make_buffer()
    buf.add(Panel("tool panel"), is_tool=True, already_printed=True)
    buf.add(Text("answer"), is_tool=False)
    buf.print_final(tools_visible=True)

    # Toggle off
    buf.rerender(tools_visible=False)
    # Toggle back on
    buf.rerender(tools_visible=True)
    full_output = out.getvalue()
    # The last rerender should include both
    # Find the last ANSI erase and check content after it
    last_erase = full_output.rfind("\033[J")
    after_last = full_output[last_erase:]
    assert "tool panel" in after_last
    assert "answer" in after_last


def test_empty_buffer():
    buf, out = _make_buffer()
    # Should not crash
    buf.print_final(tools_visible=True)
    assert buf._printed_lines == 0
    buf.rerender(tools_visible=False)
    assert out.getvalue() == ""


def test_clear_resets():
    buf, _ = _make_buffer()
    buf.add(Text("stuff"), is_tool=False)
    buf.print_final(tools_visible=True)
    assert len(buf._items) > 0
    assert buf._printed_lines > 0
    buf.clear()
    assert len(buf._items) == 0
    assert buf._printed_lines == 0
