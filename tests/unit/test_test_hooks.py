"""Tests for Feature 2: Test-Driven Self-Correction."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.hooks import HookRegistry, PostToolUse, TurnEnd, TurnStart
from forge.agent.permissions import PermissionPolicy
from forge.agent.session import (
    _detect_test_command,
    _scope_test_command,
    wire_test_hooks,
)
from forge.config import settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def deps(tmp_path):
    """AgentDeps with test-related fields."""
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
    """deps + registry with test hooks wired."""
    deps.hook_registry = registry
    wire_test_hooks(registry, deps)
    return deps, registry


# ---------------------------------------------------------------------------
# _detect_test_command
# ---------------------------------------------------------------------------


class TestDetectTestCommand:
    @pytest.mark.asyncio
    async def test_detects_pytest(self, deps, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.pytest]\n")
        result = await _detect_test_command(deps)
        assert result is not None
        assert "pytest" in result

    @pytest.mark.asyncio
    async def test_detects_uv_pytest(self, deps, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.pytest]\n")
        (tmp_path / "uv.lock").write_text("")
        result = await _detect_test_command(deps)
        assert result is not None
        assert "uv run pytest" in result

    @pytest.mark.asyncio
    async def test_detects_npm(self, deps, tmp_path):
        (tmp_path / "package.json").write_text('{"scripts":{"test":"jest"}}')
        result = await _detect_test_command(deps)
        assert result == "npm test"

    @pytest.mark.asyncio
    async def test_detects_cargo(self, deps, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\n")
        result = await _detect_test_command(deps)
        assert result == "cargo test"

    @pytest.mark.asyncio
    async def test_detects_go(self, deps, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/test\n")
        result = await _detect_test_command(deps)
        assert result == "go test ./..."

    @pytest.mark.asyncio
    async def test_no_test_runner(self, deps, tmp_path):
        result = await _detect_test_command(deps)
        assert result is None

    @pytest.mark.asyncio
    async def test_caches_result(self, deps, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.pytest]\n")
        r1 = await _detect_test_command(deps)
        r2 = await _detect_test_command(deps)
        assert r1 == r2
        assert deps.test_command is not None

    @pytest.mark.asyncio
    async def test_caches_none_result(self, deps):
        r1 = await _detect_test_command(deps)
        assert r1 is None
        assert deps._test_command_searched is True
        # Second call returns None without re-searching
        r2 = await _detect_test_command(deps)
        assert r2 is None


# ---------------------------------------------------------------------------
# _scope_test_command
# ---------------------------------------------------------------------------


class TestScopeTestCommand:
    def test_no_files_returns_base(self):
        assert _scope_test_command("pytest -x", []) == "pytest -x"

    def test_test_files_passed_directly(self, tmp_path):
        test_file = str(tmp_path / "test_foo.py")
        Path(test_file).touch()
        result = _scope_test_command("pytest -x", [test_file])
        assert test_file in result

    def test_finds_sibling_test(self, tmp_path):
        src = tmp_path / "foo.py"
        src.write_text("pass")
        test = tmp_path / "test_foo.py"
        test.write_text("pass")
        result = _scope_test_command("pytest -x", [str(src)])
        assert "test_foo.py" in result

    def test_finds_project_level_tests_dir(self, tmp_path):
        # Create a project structure
        (tmp_path / "pyproject.toml").write_text("")
        src_dir = tmp_path / "src" / "mylib"
        src_dir.mkdir(parents=True)
        src_file = src_dir / "config.py"
        src_file.write_text("pass")

        test_dir = tmp_path / "tests" / "unit"
        test_dir.mkdir(parents=True)
        test_file = test_dir / "test_config.py"
        test_file.write_text("pass")

        result = _scope_test_command("pytest -x", [str(src_file)])
        assert "test_config.py" in result

    def test_non_pytest_returns_base(self):
        result = _scope_test_command("npm test", ["/some/file.js"])
        assert result == "npm test"

    def test_deduplicates_test_files(self, tmp_path):
        # Same test file found via multiple paths
        (tmp_path / "pyproject.toml").write_text("")
        test = tmp_path / "test_foo.py"
        test.write_text("pass")
        src = tmp_path / "foo.py"
        src.write_text("pass")

        result = _scope_test_command("pytest -x", [str(src), str(src)])
        # Should only appear once
        assert result.count("test_foo.py") == 1


# ---------------------------------------------------------------------------
# wire_test_hooks — file tracking
# ---------------------------------------------------------------------------


class TestTestHooksFileTracking:
    @pytest.mark.asyncio
    async def test_tracks_write_file(self, wired):
        deps, registry = wired
        event = PostToolUse(
            tool_name="write_file",
            args={"file_path": "foo.py", "content": "pass"},
            result="ok",
            elapsed=0.1,
        )
        await registry.emit(event)
        assert len(deps._files_modified_this_turn) == 1

    @pytest.mark.asyncio
    async def test_tracks_edit_file(self, wired):
        deps, registry = wired
        event = PostToolUse(
            tool_name="edit_file",
            args={"file_path": "bar.py", "old_text": "x", "new_text": "y"},
            result="ok",
            elapsed=0.1,
        )
        await registry.emit(event)
        assert len(deps._files_modified_this_turn) == 1

    @pytest.mark.asyncio
    async def test_tracks_write_commands(self, wired):
        deps, registry = wired
        event = PostToolUse(
            tool_name="run_command",
            args={"command": "echo hello > output.txt"},
            result="ok",
            elapsed=0.1,
        )
        await registry.emit(event)
        assert len(deps._files_modified_this_turn) == 1
        assert deps._files_modified_this_turn[0].startswith("__cmd_write:")

    @pytest.mark.asyncio
    async def test_ignores_read_tools(self, wired):
        deps, registry = wired
        event = PostToolUse(
            tool_name="read_file",
            args={"file_path": "foo.py"},
            result="contents",
            elapsed=0.1,
        )
        await registry.emit(event)
        assert len(deps._files_modified_this_turn) == 0

    @pytest.mark.asyncio
    async def test_clears_on_turn_start(self, wired):
        deps, registry = wired
        deps._files_modified_this_turn = ["foo.py", "bar.py"]
        await registry.emit(TurnStart(turn_number=1, prompt="test"))
        assert deps._files_modified_this_turn == []

    @pytest.mark.asyncio
    async def test_clears_on_turn_end(self, wired):
        deps, registry = wired
        deps._files_modified_this_turn = ["foo.py", "bar.py"]
        await registry.emit(TurnEnd(
            turn_number=1, tool_call_count=1, elapsed=1.0,
            tokens_in=0, tokens_out=0,
        ))
        assert deps._files_modified_this_turn == []


# ---------------------------------------------------------------------------
# wire_test_hooks — test execution
# ---------------------------------------------------------------------------


class TestTestHooksExecution:
    @pytest.mark.asyncio
    async def test_runs_tests_on_turn_end(self, wired, tmp_path):
        deps, registry = wired
        # Create a failing test
        (tmp_path / "pyproject.toml").write_text("")
        deps._files_modified_this_turn = [str(tmp_path / "foo.py")]

        # Mock subprocess to simulate test failure
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "FAILED test_foo.py::test_bar - AssertionError"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            await registry.emit(TurnEnd(
                turn_number=1, tool_call_count=1, elapsed=1.0,
                tokens_in=0, tokens_out=0,
            ))

        assert deps.test_results is not None
        assert "FAILED" in deps.test_results

    @pytest.mark.asyncio
    async def test_no_tests_when_disabled(self, deps, tmp_path):
        registry = HookRegistry()
        deps.hook_registry = registry
        wire_test_hooks(registry, deps)

        (tmp_path / "pyproject.toml").write_text("")
        deps._files_modified_this_turn = [str(tmp_path / "foo.py")]

        with patch.object(settings.agent, "test_enabled", False):
            with patch("subprocess.run") as mock_run:
                await registry.emit(TurnEnd(
                    turn_number=1, tool_call_count=1, elapsed=1.0,
                    tokens_in=0, tokens_out=0,
                ))
                mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_clears_results_on_success(self, wired, tmp_path):
        deps, registry = wired
        (tmp_path / "pyproject.toml").write_text("")
        deps._files_modified_this_turn = [str(tmp_path / "foo.py")]

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "1 passed"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            await registry.emit(TurnEnd(
                turn_number=1, tool_call_count=1, elapsed=1.0,
                tokens_in=0, tokens_out=0,
            ))

        assert deps.test_results is None

    @pytest.mark.asyncio
    async def test_handles_timeout(self, wired, tmp_path):
        deps, registry = wired
        (tmp_path / "pyproject.toml").write_text("")
        deps._files_modified_this_turn = [str(tmp_path / "foo.py")]

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)):
            await registry.emit(TurnEnd(
                turn_number=1, tool_call_count=1, elapsed=1.0,
                tokens_in=0, tokens_out=0,
            ))

        assert deps.test_results is not None
        assert "timed out" in deps.test_results

    @pytest.mark.asyncio
    async def test_skips_below_min_writes(self, wired, tmp_path):
        deps, registry = wired
        (tmp_path / "pyproject.toml").write_text("")
        # No files modified
        deps._files_modified_this_turn = []

        with patch("subprocess.run") as mock_run:
            await registry.emit(TurnEnd(
                turn_number=1, tool_call_count=1, elapsed=1.0,
                tokens_in=0, tokens_out=0,
            ))
            mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_truncates_long_output(self, wired, tmp_path):
        deps, registry = wired
        (tmp_path / "pyproject.toml").write_text("")
        deps._files_modified_this_turn = [str(tmp_path / "foo.py")]

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "X" * 5000
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            await registry.emit(TurnEnd(
                turn_number=1, tool_call_count=1, elapsed=1.0,
                tokens_in=0, tokens_out=0,
            ))

        assert deps.test_results is not None
        assert "truncated" in deps.test_results
        assert len(deps.test_results) < 5000


# ---------------------------------------------------------------------------
# AgentDeps fields
# ---------------------------------------------------------------------------


class TestDepsFields:
    def test_default_test_fields(self):
        deps = AgentDeps(cwd=Path("/tmp"))
        assert deps.test_results is None
        assert deps.test_command is None
        assert deps._test_command_searched is False
        assert deps._files_modified_this_turn == []
        assert deps.critique_results is None


# ---------------------------------------------------------------------------
# Config settings
# ---------------------------------------------------------------------------


class TestConfigSettings:
    def test_test_enabled_default(self):
        assert settings.agent.test_enabled is True

    def test_test_timeout_default(self):
        assert settings.agent.test_timeout == 30.0

    def test_test_min_writes_default(self):
        assert settings.agent.test_min_writes == 1
