"""Tests for .forge/test-config.json support in _detect_test_command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.permissions import PermissionPolicy
from forge.agent.session import _detect_test_command


@pytest.fixture
def deps(tmp_path):
    return AgentDeps(
        cwd=tmp_path,
        console=Console(file=None, force_terminal=False, no_color=True),
        permission=PermissionPolicy.YOLO,
    )


class TestDetectTestCommand:
    @pytest.mark.asyncio
    async def test_forge_test_config(self, deps, tmp_path):
        """Should pick up .forge/test-config.json testCommand."""
        config_dir = tmp_path / ".forge"
        config_dir.mkdir()
        (config_dir / "test-config.json").write_text(
            json.dumps({"testCommand": "echo ok"})
        )

        result = await _detect_test_command(deps)
        assert result == "echo ok"
        assert deps.test_command == "echo ok"

    @pytest.mark.asyncio
    async def test_claude_test_config(self, deps, tmp_path):
        """Should pick up .claude/test-config.json as fallback."""
        config_dir = tmp_path / ".claude"
        config_dir.mkdir()
        (config_dir / "test-config.json").write_text(
            json.dumps({"testCommand": "npm test"})
        )

        result = await _detect_test_command(deps)
        assert result == "npm test"

    @pytest.mark.asyncio
    async def test_forge_config_takes_priority(self, deps, tmp_path):
        """Forge config should be checked before Claude config."""
        for name in (".forge", ".claude"):
            d = tmp_path / name
            d.mkdir()
            (d / "test-config.json").write_text(
                json.dumps({"testCommand": f"{name}-cmd"})
            )

        result = await _detect_test_command(deps)
        assert result == ".forge-cmd"

    @pytest.mark.asyncio
    async def test_invalid_json_falls_through(self, deps, tmp_path):
        """Invalid JSON should fall through to heuristic detection."""
        config_dir = tmp_path / ".forge"
        config_dir.mkdir()
        (config_dir / "test-config.json").write_text("not json")

        # No pyproject.toml etc., so should return None
        result = await _detect_test_command(deps)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_test_command_falls_through(self, deps, tmp_path):
        """Empty testCommand should fall through."""
        config_dir = tmp_path / ".forge"
        config_dir.mkdir()
        (config_dir / "test-config.json").write_text(
            json.dumps({"testCommand": ""})
        )

        result = await _detect_test_command(deps)
        assert result is None

    @pytest.mark.asyncio
    async def test_caches_result(self, deps, tmp_path):
        """Result should be cached after first detection."""
        config_dir = tmp_path / ".forge"
        config_dir.mkdir()
        (config_dir / "test-config.json").write_text(
            json.dumps({"testCommand": "pytest"})
        )

        result1 = await _detect_test_command(deps)
        assert result1 == "pytest"

        # Remove config — should still return cached value
        (config_dir / "test-config.json").unlink()
        result2 = await _detect_test_command(deps)
        assert result2 == "pytest"
