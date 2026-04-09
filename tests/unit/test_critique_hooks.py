"""Tests for Feature 4: Critique-Before-Commit."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.hooks import HookRegistry, PostToolUse, TurnEnd, TurnStart
from forge.agent.permissions import PermissionPolicy
from forge.agent.session import (
    _call_critique_model,
    wire_critique_hooks,
    wire_test_hooks,
)
from forge.config import settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
def wired(deps, registry):
    """deps + registry with critique hooks wired."""
    deps.hook_registry = registry
    wire_critique_hooks(registry, deps)
    return deps, registry


# ---------------------------------------------------------------------------
# wire_critique_hooks
# ---------------------------------------------------------------------------


class TestCritiqueHooksExecution:
    @pytest.mark.asyncio
    async def test_runs_critique_on_multi_file_changes(self, wired, tmp_path):
        deps, registry = wired
        # Simulate 2+ files modified
        deps._files_modified_this_turn = [
            str(tmp_path / "a.py"),
            str(tmp_path / "b.py"),
        ]

        # Mock git diff
        mock_diff = MagicMock()
        mock_diff.stdout = "diff --git a/a.py b/a.py\n+new line"
        mock_diff.returncode = 0

        # Mock critique model response
        with patch("subprocess.run", return_value=mock_diff):
            with patch(
                "forge.agent.session._call_critique_model",
                new_callable=AsyncMock,
                return_value=("Issue: missing error handling in a.py", "gemini"),
            ):
                await registry.emit(TurnEnd(
                    turn_number=1, tool_call_count=2, elapsed=1.0,
                    tokens_in=0, tokens_out=0,
                ))

        assert deps.critique_results is not None
        assert "Issue" in deps.critique_results

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self, deps, tmp_path):
        registry = HookRegistry()
        deps.hook_registry = registry
        wire_critique_hooks(registry, deps)

        deps._files_modified_this_turn = [
            str(tmp_path / "a.py"),
            str(tmp_path / "b.py"),
        ]

        with patch.object(settings.agent, "critique_enabled", False):
            with patch("subprocess.run") as mock_run:
                await registry.emit(TurnEnd(
                    turn_number=1, tool_call_count=2, elapsed=1.0,
                    tokens_in=0, tokens_out=0,
                ))
                mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_below_min_writes(self, wired, tmp_path):
        deps, registry = wired
        # Only 1 file modified, default min is 2
        deps._files_modified_this_turn = [str(tmp_path / "a.py")]

        with patch("subprocess.run") as mock_run:
            await registry.emit(TurnEnd(
                turn_number=1, tool_call_count=1, elapsed=1.0,
                tokens_in=0, tokens_out=0,
            ))
            mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_no_diff(self, wired, tmp_path):
        deps, registry = wired
        deps._files_modified_this_turn = [
            str(tmp_path / "a.py"),
            str(tmp_path / "b.py"),
        ]

        # Git diff returns empty
        mock_diff = MagicMock()
        mock_diff.stdout = ""
        mock_diff.returncode = 0

        with patch("subprocess.run", return_value=mock_diff):
            with patch(
                "forge.agent.session._call_critique_model",
                new_callable=AsyncMock,
            ) as mock_critique:
                await registry.emit(TurnEnd(
                    turn_number=1, tool_call_count=2, elapsed=1.0,
                    tokens_in=0, tokens_out=0,
                ))
                mock_critique.assert_not_called()

    @pytest.mark.asyncio
    async def test_lgtm_not_stored(self, wired, tmp_path):
        deps, registry = wired
        deps._files_modified_this_turn = [
            str(tmp_path / "a.py"),
            str(tmp_path / "b.py"),
        ]

        mock_diff = MagicMock()
        mock_diff.stdout = "diff --git a/a.py b/a.py\n+good code"
        mock_diff.returncode = 0

        with patch("subprocess.run", return_value=mock_diff):
            with patch(
                "forge.agent.session._call_critique_model",
                new_callable=AsyncMock,
                return_value=("LGTM - looks good to me", "gemini"),
            ):
                await registry.emit(TurnEnd(
                    turn_number=1, tool_call_count=2, elapsed=1.0,
                    tokens_in=0, tokens_out=0,
                ))

        assert deps.critique_results is None

    @pytest.mark.asyncio
    async def test_truncates_large_diff(self, wired, tmp_path):
        deps, registry = wired
        deps._files_modified_this_turn = [
            str(tmp_path / "a.py"),
            str(tmp_path / "b.py"),
        ]

        mock_diff = MagicMock()
        mock_diff.stdout = "X" * 20000  # Much larger than default max
        mock_diff.returncode = 0

        captured_diff = []

        async def capture_critique(diff_text, deps):
            captured_diff.append(diff_text)
            return "LGTM", "local"

        with patch("subprocess.run", return_value=mock_diff):
            with patch(
                "forge.agent.session._call_critique_model",
                side_effect=capture_critique,
            ):
                await registry.emit(TurnEnd(
                    turn_number=1, tool_call_count=2, elapsed=1.0,
                    tokens_in=0, tokens_out=0,
                ))

        assert len(captured_diff) == 1
        assert len(captured_diff[0]) <= settings.agent.critique_max_diff_chars + 50

    @pytest.mark.asyncio
    async def test_handles_git_not_found(self, wired, tmp_path):
        deps, registry = wired
        deps._files_modified_this_turn = [
            str(tmp_path / "a.py"),
            str(tmp_path / "b.py"),
        ]

        with patch("subprocess.run", side_effect=FileNotFoundError):
            await registry.emit(TurnEnd(
                turn_number=1, tool_call_count=2, elapsed=1.0,
                tokens_in=0, tokens_out=0,
            ))

        assert deps.critique_results is None

    @pytest.mark.asyncio
    async def test_ignores_cmd_write_files(self, wired, tmp_path):
        deps, registry = wired
        # One real file + one cmd_write — should only count real files
        deps._files_modified_this_turn = [
            str(tmp_path / "a.py"),
            "__cmd_write:echo hello > foo.txt",
        ]

        with patch("subprocess.run") as mock_run:
            await registry.emit(TurnEnd(
                turn_number=1, tool_call_count=1, elapsed=1.0,
                tokens_in=0, tokens_out=0,
            ))
            # Only 1 real file < min_writes(2), so should not run
            mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# _call_critique_model
# ---------------------------------------------------------------------------


class TestCallCritiqueModel:
    @pytest.mark.asyncio
    async def test_returns_critique(self, deps):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Found potential bug in line 5"}
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result, source = await _call_critique_model("diff content", deps)
            assert result == "Found potential bug in line 5"
            assert source == "local"

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self, deps):
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.side_effect = Exception("connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result, source = await _call_critique_model("diff", deps)
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_non_200(self, deps):
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result, source = await _call_critique_model("diff", deps)
            assert result is None


# ---------------------------------------------------------------------------
# Config settings
# ---------------------------------------------------------------------------


class TestCritiqueConfig:
    def test_critique_enabled_default(self):
        assert settings.agent.critique_enabled is True

    def test_critique_min_writes_default(self):
        assert settings.agent.critique_min_writes == 2

    def test_critique_max_diff_chars_default(self):
        assert settings.agent.critique_max_diff_chars == 8000


# ---------------------------------------------------------------------------
# AgentDeps critique field
# ---------------------------------------------------------------------------


class TestDepsCritiqueField:
    def test_default_is_none(self):
        deps = AgentDeps(cwd=Path("/tmp"))
        assert deps.critique_results is None
