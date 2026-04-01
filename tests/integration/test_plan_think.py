"""Integration tests for /plan and /think features."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.loop import (
    PLAN_OVERLAY,
    _maybe_prepend_think,
    _run_with_status,
)
from forge.agent.permissions import PermissionPolicy


class TestPlanOverlay:
    """Test the planning system prompt."""

    def test_plan_overlay_disables_tools(self):
        assert "Do NOT execute any actions" in PLAN_OVERLAY
        assert "Do NOT" in PLAN_OVERLAY or "do NOT" in PLAN_OVERLAY

    def test_plan_overlay_requests_structure(self):
        assert "Steps" in PLAN_OVERLAY
        assert "Files" in PLAN_OVERLAY
        assert "Goal" in PLAN_OVERLAY


class TestThinkPrepend:
    """Test /think prompt prepending."""

    def test_disabled_returns_unchanged(self):
        deps = AgentDeps(
            cwd=Path("/tmp"),
            console=Console(file=io.StringIO()),
            thinking_enabled=False,
        )
        assert _maybe_prepend_think("hello", deps) == "hello"

    def test_enabled_prepends_think_tag(self):
        deps = AgentDeps(
            cwd=Path("/tmp"),
            console=Console(file=io.StringIO()),
            thinking_enabled=True,
        )
        result = _maybe_prepend_think("hello", deps)
        assert result == "/think\nhello"

    def test_enabled_with_multiline_prompt(self):
        deps = AgentDeps(
            cwd=Path("/tmp"),
            console=Console(file=io.StringIO()),
            thinking_enabled=True,
        )
        result = _maybe_prepend_think("line1\nline2\nline3", deps)
        assert result.startswith("/think\n")
        assert "line1\nline2\nline3" in result

    def test_empty_prompt(self):
        deps = AgentDeps(
            cwd=Path("/tmp"),
            console=Console(file=io.StringIO()),
            thinking_enabled=True,
        )
        result = _maybe_prepend_think("", deps)
        assert result == "/think\n"


class TestRunWithStatus:
    """Test _run_with_status wraps agent.run with tracker lifecycle."""

    @pytest.mark.asyncio
    async def test_sets_and_clears_tracker_on_deps(self):
        """Tracker should be set on deps during run and cleared after."""
        console = Console(file=io.StringIO())
        deps = AgentDeps(cwd=Path("/tmp"), console=console)

        # Track what tracker was set during agent.run
        captured_tracker = None

        async def mock_run(prompt, **kwargs):
            nonlocal captured_tracker
            captured_tracker = kwargs["deps"].status_tracker
            result = MagicMock()
            result.all_messages.return_value = []
            return result

        agent = MagicMock()
        agent.run = mock_run

        await _run_with_status(agent, "test", deps, None)

        # Tracker was set during run
        assert captured_tracker is not None
        # Tracker is cleared after run
        assert deps.status_tracker is None

    @pytest.mark.asyncio
    async def test_returns_message_history(self):
        """Should return result.all_messages()."""
        console = Console(file=io.StringIO())
        deps = AgentDeps(cwd=Path("/tmp"), console=console)

        expected_messages = [MagicMock(), MagicMock()]

        async def mock_run(prompt, **kwargs):
            result = MagicMock()
            result.all_messages.return_value = expected_messages
            return result

        agent = MagicMock()
        agent.run = mock_run

        result = await _run_with_status(agent, "test", deps, None)
        assert result == expected_messages

    @pytest.mark.asyncio
    async def test_clears_tracker_on_error(self):
        """Tracker should be cleared even if agent.run raises."""
        console = Console(file=io.StringIO())
        deps = AgentDeps(cwd=Path("/tmp"), console=console)

        async def mock_run(prompt, **kwargs):
            raise RuntimeError("model error")

        agent = MagicMock()
        agent.run = mock_run

        with pytest.raises(RuntimeError, match="model error"):
            await _run_with_status(agent, "test", deps, None)

        # Tracker should still be cleaned up
        assert deps.status_tracker is None

    @pytest.mark.asyncio
    async def test_prints_summary(self, capsys):
        """Should print a summary after completion."""
        console = Console(force_terminal=False)
        deps = AgentDeps(cwd=Path("/tmp"), console=console)

        async def mock_run(prompt, **kwargs):
            result = MagicMock()
            result.all_messages.return_value = []
            return result

        agent = MagicMock()
        agent.run = mock_run

        await _run_with_status(agent, "test", deps, None)
        # Summary is printed via console.print (Rich markup)
        # Just verify no crash — output goes to console's file
