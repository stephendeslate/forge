"""Integration tests for the 4 architectural features working together.

Tests hook interactions, priority ordering, and feature combinations.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.hooks import HookRegistry, PostToolUse, TurnEnd, TurnStart
from forge.agent.permissions import PermissionPolicy
from forge.agent.session import wire_critique_hooks, wire_test_hooks
from forge.config import settings


@pytest.fixture
def deps(tmp_path):
    return AgentDeps(
        cwd=tmp_path,
        console=Console(file=None, force_terminal=False, no_color=True),
        permission=PermissionPolicy.YOLO,
    )


@pytest.fixture
def registry():
    return HookRegistry()


@pytest.fixture
def full_wired(deps, registry):
    """deps + registry with BOTH test and critique hooks wired."""
    deps.hook_registry = registry
    wire_test_hooks(registry, deps)
    wire_critique_hooks(registry, deps)
    return deps, registry


# ---------------------------------------------------------------------------
# Priority ordering: test(25) → critique(30) → clear(50)
# ---------------------------------------------------------------------------


class TestHookPriorityOrdering:
    @pytest.mark.asyncio
    async def test_critique_sees_files_before_clear(self, full_wired, tmp_path):
        """Critique hook at priority 30 should see _files_modified_this_turn
        before it's cleared at priority 50."""
        deps, registry = full_wired
        (tmp_path / "pyproject.toml").write_text("")
        deps._files_modified_this_turn = [
            str(tmp_path / "a.py"),
            str(tmp_path / "b.py"),
        ]

        critique_saw_files = []

        # Mock test run (succeeds)
        mock_test = MagicMock()
        mock_test.returncode = 0
        mock_test.stdout = "1 passed"
        mock_test.stderr = ""

        # Mock git diff
        mock_diff = MagicMock()
        mock_diff.stdout = "diff --git a/a.py"
        mock_diff.returncode = 0

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, str) and "pytest" in cmd:
                return mock_test
            # git diff call
            return mock_diff

        async def capture_critique(diff_text, deps):
            critique_saw_files.append(len(deps._files_modified_this_turn))
            return "LGTM"

        with patch("subprocess.run", side_effect=capture_run):
            with patch(
                "forge.agent.session._call_critique_model",
                side_effect=capture_critique,
            ):
                await registry.emit(TurnEnd(
                    turn_number=1, tool_call_count=2, elapsed=1.0,
                    tokens_in=0, tokens_out=0,
                ))

        # Critique should have seen files (before clear at 50)
        assert len(critique_saw_files) == 1
        assert critique_saw_files[0] == 2

        # After TurnEnd, files should be cleared
        assert deps._files_modified_this_turn == []

    @pytest.mark.asyncio
    async def test_turn_start_clears_files(self, full_wired):
        deps, registry = full_wired
        deps._files_modified_this_turn = ["a.py", "b.py"]

        await registry.emit(TurnStart(turn_number=2, prompt="next turn"))
        assert deps._files_modified_this_turn == []


# ---------------------------------------------------------------------------
# File tracking → test + critique pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_track_then_test_then_critique(self, full_wired, tmp_path):
        """Full pipeline: track file writes → run tests → run critique."""
        deps, registry = full_wired
        (tmp_path / "pyproject.toml").write_text("")

        # Track 2 file writes
        for name in ("a.py", "b.py"):
            event = PostToolUse(
                tool_name="write_file",
                args={"file_path": str(tmp_path / name), "content": "pass"},
                result="ok",
                elapsed=0.1,
            )
            await registry.emit(event)

        assert len(deps._files_modified_this_turn) == 2

        # Mock test (passes) + critique
        mock_test = MagicMock()
        mock_test.returncode = 0
        mock_test.stdout = "2 passed"
        mock_test.stderr = ""

        mock_diff = MagicMock()
        mock_diff.stdout = "diff --git a/a.py"
        mock_diff.returncode = 0

        def mock_run(cmd, **kwargs):
            if isinstance(cmd, str) and "pytest" in cmd:
                return mock_test
            return mock_diff

        with patch("subprocess.run", side_effect=mock_run):
            with patch(
                "forge.agent.session._call_critique_model",
                new_callable=AsyncMock,
                return_value="Missing docstrings in a.py",
            ):
                await registry.emit(TurnEnd(
                    turn_number=1, tool_call_count=2, elapsed=1.0,
                    tokens_in=0, tokens_out=0,
                ))

        # Tests passed, so no test_results
        assert deps.test_results is None
        # Critique found issues
        assert deps.critique_results is not None
        assert "Missing docstrings" in deps.critique_results
        # Files cleared after turn end
        assert deps._files_modified_this_turn == []

    @pytest.mark.asyncio
    async def test_test_failure_still_allows_critique(self, full_wired, tmp_path):
        """Both test failures and critique should be captured."""
        deps, registry = full_wired
        (tmp_path / "pyproject.toml").write_text("")
        deps._files_modified_this_turn = [
            str(tmp_path / "a.py"),
            str(tmp_path / "b.py"),
        ]

        mock_test = MagicMock()
        mock_test.returncode = 1
        mock_test.stdout = "FAILED test_a.py"
        mock_test.stderr = ""

        mock_diff = MagicMock()
        mock_diff.stdout = "diff --git a/a.py"
        mock_diff.returncode = 0

        def mock_run(cmd, **kwargs):
            if isinstance(cmd, str) and "pytest" in cmd:
                return mock_test
            return mock_diff

        with patch("subprocess.run", side_effect=mock_run):
            with patch(
                "forge.agent.session._call_critique_model",
                new_callable=AsyncMock,
                return_value="Bug: off-by-one in loop",
            ):
                await registry.emit(TurnEnd(
                    turn_number=1, tool_call_count=2, elapsed=1.0,
                    tokens_in=0, tokens_out=0,
                ))

        assert deps.test_results is not None
        assert "FAILED" in deps.test_results
        assert deps.critique_results is not None
        assert "Bug" in deps.critique_results


# ---------------------------------------------------------------------------
# Feature gating via config
# ---------------------------------------------------------------------------


class TestFeatureGating:
    @pytest.mark.asyncio
    async def test_both_disabled(self, full_wired, tmp_path):
        deps, registry = full_wired
        (tmp_path / "pyproject.toml").write_text("")
        deps._files_modified_this_turn = [
            str(tmp_path / "a.py"),
            str(tmp_path / "b.py"),
        ]

        with patch.object(settings.agent, "test_enabled", False):
            with patch.object(settings.agent, "critique_enabled", False):
                with patch("subprocess.run") as mock_run:
                    await registry.emit(TurnEnd(
                        turn_number=1, tool_call_count=2, elapsed=1.0,
                        tokens_in=0, tokens_out=0,
                    ))
                    mock_run.assert_not_called()

        assert deps.test_results is None
        assert deps.critique_results is None

    @pytest.mark.asyncio
    async def test_edit_file_tracked(self, full_wired, tmp_path):
        """Edit file events should also be tracked for both features."""
        deps, registry = full_wired

        event = PostToolUse(
            tool_name="edit_file",
            args={"file_path": str(tmp_path / "x.py"), "old_text": "a", "new_text": "b"},
            result="ok",
            elapsed=0.1,
        )
        await registry.emit(event)
        assert len(deps._files_modified_this_turn) == 1


# ---------------------------------------------------------------------------
# Evidence-grounded planning tools integration
# ---------------------------------------------------------------------------


class TestReadOnlyToolsIntegration:
    def test_read_only_tools_are_subset(self):
        from forge.agent.tools import ALL_TOOLS, READ_ONLY_TOOLS

        all_names = {t.name for t in ALL_TOOLS}
        ro_names = {t.name for t in READ_ONLY_TOOLS}
        # All read-only tools should exist in ALL_TOOLS (analyze_impact may be extra)
        assert ro_names.issubset(all_names | {"analyze_impact"})

    def test_plan_overlay_content(self):
        from forge.agent.loop import PLAN_OVERLAY

        assert "read-only" in PLAN_OVERLAY.lower() or "read only" in PLAN_OVERLAY.lower()
        assert "Goal" in PLAN_OVERLAY
        assert "Steps" in PLAN_OVERLAY
