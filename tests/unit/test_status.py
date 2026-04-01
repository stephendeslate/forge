"""Tests for the StatusTracker and Phase enum."""

from __future__ import annotations

import asyncio
import io
import time

import pytest
from rich.console import Console

from forge.agent.status import Phase, StatusTracker


class TestPhase:
    def test_all_phases_have_values(self):
        assert Phase.THINKING.value == "thinking"
        assert Phase.STREAMING.value == "streaming"
        assert Phase.TOOL_CALL.value == "tool call"
        assert Phase.TOOL_EXEC.value == "executing"
        assert Phase.DONE.value == "done"


class TestStatusTrackerLifecycle:
    def test_start_sets_active(self):
        tracker = StatusTracker(console=Console(file=io.StringIO()))
        assert not tracker._active
        tracker.start()
        assert tracker._active
        tracker.stop()

    def test_stop_clears_active(self):
        tracker = StatusTracker(console=Console(file=io.StringIO()))
        tracker.start()
        tracker.stop()
        assert not tracker._active

    def test_start_records_time(self):
        tracker = StatusTracker(console=Console(file=io.StringIO()))
        before = time.monotonic()
        tracker.start()
        after = time.monotonic()
        assert before <= tracker._start_time <= after
        tracker.stop()

    def test_double_stop_is_safe(self):
        tracker = StatusTracker(console=Console(file=io.StringIO()))
        tracker.start()
        tracker.stop()
        tracker.stop()  # Should not raise


class TestStatusTrackerPhases:
    def test_set_phase(self):
        tracker = StatusTracker(console=Console(file=io.StringIO()))
        tracker.start()
        tracker.set_phase(Phase.STREAMING)
        assert tracker._phase == Phase.STREAMING
        tracker.set_phase(Phase.TOOL_CALL, "read_file")
        assert tracker._phase == Phase.TOOL_CALL
        assert tracker._detail == "read_file"
        tracker.stop()

    def test_pause_and_resume(self):
        tracker = StatusTracker(console=Console(file=io.StringIO()))
        tracker.start()
        assert not tracker._paused
        tracker.pause()
        assert tracker._paused
        tracker.resume()
        assert not tracker._paused
        tracker.stop()

    def test_pause_before_start_is_safe(self):
        tracker = StatusTracker(console=Console(file=io.StringIO()))
        tracker.pause()  # _active is False, should be no-op
        tracker.resume()


class TestStatusTrackerToolCounting:
    def test_initial_count_is_zero(self):
        tracker = StatusTracker(console=Console(file=io.StringIO()))
        assert tracker.tool_calls == 0

    def test_increment(self):
        tracker = StatusTracker(console=Console(file=io.StringIO()))
        tracker.increment_tool_calls()
        tracker.increment_tool_calls()
        assert tracker.tool_calls == 2

    def test_increment_many(self):
        tracker = StatusTracker(console=Console(file=io.StringIO()))
        for _ in range(10):
            tracker.increment_tool_calls()
        assert tracker.tool_calls == 10


class TestStatusTrackerSummary:
    def test_summary_no_tools(self):
        tracker = StatusTracker(console=Console(file=io.StringIO()))
        tracker.start()
        summary = tracker.summary()
        tracker.stop()
        # Should contain elapsed time, no tool calls
        assert "[dim]" in summary
        assert "s[/dim]" in summary
        assert "tool" not in summary

    def test_summary_one_tool(self):
        tracker = StatusTracker(console=Console(file=io.StringIO()))
        tracker.start()
        tracker.increment_tool_calls()
        summary = tracker.summary()
        tracker.stop()
        assert "1 tool call in" in summary
        # Singular "call" not "calls"
        assert "calls" not in summary

    def test_summary_multiple_tools(self):
        tracker = StatusTracker(console=Console(file=io.StringIO()))
        tracker.start()
        for _ in range(3):
            tracker.increment_tool_calls()
        summary = tracker.summary()
        tracker.stop()
        assert "3 tool calls in" in summary


class TestStatusTrackerTerminalOutput:
    def test_non_terminal_skips_print(self):
        """Non-terminal console should not write ANSI sequences."""
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, no_color=True)
        tracker = StatusTracker(console=console)
        tracker.start()
        tracker.set_phase(Phase.THINKING)
        tracker._print_status()  # Should be a no-op for non-terminal
        tracker.stop()

    def test_terminal_writes_status(self, capsys):
        """Terminal console writes status to stderr."""
        console = Console(force_terminal=True)
        tracker = StatusTracker(console=console)
        tracker.start()
        tracker.set_phase(Phase.THINKING)
        tracker._print_status()
        tracker.stop()
        captured = capsys.readouterr()
        # Status goes to stderr
        assert "thinking" in captured.err


class TestStatusTrackerAsync:
    @pytest.mark.asyncio
    async def test_ticker_runs_and_stops(self):
        """The async ticker task should run and be cancellable."""
        console = Console(force_terminal=True)
        tracker = StatusTracker(console=console)
        tracker.start()

        # Ticker should be running
        assert tracker._ticker is not None
        assert not tracker._ticker.done()

        # Let it tick a couple times
        await asyncio.sleep(0.25)

        tracker.stop()
        # After stop, ticker should be cancelled
        assert tracker._ticker is None

    @pytest.mark.asyncio
    async def test_elapsed_time_increases(self):
        tracker = StatusTracker(console=Console(file=io.StringIO()))
        tracker.start()
        await asyncio.sleep(0.15)
        elapsed = tracker._elapsed()
        tracker.stop()
        # Should be at least 0.1s
        val = float(elapsed.rstrip("s"))
        assert val >= 0.1
