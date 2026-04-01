"""Integration tests for StatusTracker with render and agent deps."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.permissions import PermissionPolicy
from forge.agent.status import Phase, StatusTracker


class TestStatusTrackerWithDeps:
    """Test StatusTracker integrates correctly with AgentDeps."""

    def test_deps_default_no_tracker(self):
        deps = AgentDeps(cwd=Path("/tmp"), console=Console(file=io.StringIO()))
        assert deps.status_tracker is None

    def test_deps_with_tracker(self):
        console = Console(file=io.StringIO())
        tracker = StatusTracker(console=console)
        deps = AgentDeps(
            cwd=Path("/tmp"),
            console=console,
            status_tracker=tracker,
        )
        assert deps.status_tracker is tracker

    def test_deps_thinking_default_off(self):
        deps = AgentDeps(cwd=Path("/tmp"), console=Console(file=io.StringIO()))
        assert deps.thinking_enabled is False

    def test_deps_thinking_toggle(self):
        deps = AgentDeps(cwd=Path("/tmp"), console=Console(file=io.StringIO()))
        deps.thinking_enabled = True
        assert deps.thinking_enabled is True
        deps.thinking_enabled = False
        assert deps.thinking_enabled is False


class TestStatusTrackerPhaseTransitions:
    """Test realistic phase transition sequences."""

    @pytest.mark.asyncio
    async def test_typical_text_only_flow(self):
        """THINKING → STREAMING → DONE (no tools)."""
        console = Console(force_terminal=True)
        tracker = StatusTracker(console=console)
        tracker.start()

        tracker.set_phase(Phase.THINKING)
        assert tracker._phase == Phase.THINKING

        tracker.set_phase(Phase.STREAMING)
        tracker.pause()
        assert tracker._paused
        # Simulate streaming...
        await asyncio.sleep(0.05)
        tracker.resume()
        assert not tracker._paused

        tracker.set_phase(Phase.DONE)
        summary = tracker.summary()
        tracker.stop()

        assert "tool" not in summary  # No tools used
        assert "s" in summary  # Has elapsed time

    @pytest.mark.asyncio
    async def test_typical_tool_use_flow(self):
        """THINKING → TOOL_CALL → TOOL_EXEC → THINKING → STREAMING → DONE."""
        console = Console(force_terminal=True)
        tracker = StatusTracker(console=console)
        tracker.start()

        tracker.set_phase(Phase.THINKING)
        tracker.set_phase(Phase.TOOL_CALL, "read_file")
        tracker.increment_tool_calls()
        tracker.set_phase(Phase.TOOL_EXEC, "read_file")
        tracker.set_phase(Phase.THINKING)
        tracker.set_phase(Phase.STREAMING)
        tracker.pause()
        await asyncio.sleep(0.05)
        tracker.resume()
        tracker.set_phase(Phase.DONE)

        summary = tracker.summary()
        tracker.stop()

        assert "1 tool call" in summary
        assert "calls" not in summary  # Singular

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_flow(self):
        """Multiple tool calls in sequence."""
        console = Console(force_terminal=True)
        tracker = StatusTracker(console=console)
        tracker.start()

        for tool in ["read_file", "edit_file", "run_command"]:
            tracker.set_phase(Phase.TOOL_CALL, tool)
            tracker.increment_tool_calls()
            tracker.set_phase(Phase.TOOL_EXEC, tool)
            tracker.set_phase(Phase.THINKING)

        tracker.set_phase(Phase.DONE)
        summary = tracker.summary()
        tracker.stop()

        assert "3 tool calls" in summary


class TestStatusTrackerCancellation:
    """Test interrupt/cancellation scenarios."""

    @pytest.mark.asyncio
    async def test_stop_during_tick(self):
        """Stopping while ticker is active should not raise."""
        console = Console(force_terminal=True)
        tracker = StatusTracker(console=console)
        tracker.start()
        await asyncio.sleep(0.15)  # Let ticker run a few cycles
        tracker.stop()
        # Should complete cleanly
        assert not tracker._active

    @pytest.mark.asyncio
    async def test_stop_while_paused(self):
        """Stopping while paused should clean up."""
        console = Console(force_terminal=True)
        tracker = StatusTracker(console=console)
        tracker.start()
        tracker.pause()
        tracker.stop()
        assert not tracker._active
