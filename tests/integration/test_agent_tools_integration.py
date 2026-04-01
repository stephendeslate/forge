"""Integration tests — agent tools with permission policies."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.permissions import PermissionPolicy
from forge.agent.tools import read_file, write_file, edit_file, run_command


@pytest.fixture
def auto_ctx(tmp_path):
    """RunContext with AUTO permissions."""
    mock = MagicMock()
    mock.deps = AgentDeps(
        cwd=tmp_path,
        console=Console(file=None, force_terminal=False, no_color=True),
        permission=PermissionPolicy.AUTO,
    )
    return mock


@pytest.fixture
def ask_ctx(tmp_path):
    """RunContext with ASK permissions."""
    mock = MagicMock()
    mock.deps = AgentDeps(
        cwd=tmp_path,
        console=Console(file=None, force_terminal=False, no_color=True),
        permission=PermissionPolicy.ASK,
    )
    return mock


class TestAutoPermissions:
    async def test_read_allowed_without_prompt(self, auto_ctx, tmp_path):
        (tmp_path / "test.txt").write_text("content")
        result = await read_file(auto_ctx, "test.txt")
        assert "content" in result

    async def test_write_denied_when_user_says_no(self, auto_ctx, tmp_path):
        with patch("forge.agent.permissions._prompt_user", new_callable=AsyncMock, return_value=False):
            result = await write_file(auto_ctx, "test.txt", "data")
            assert "denied" in result.lower()
            assert not (tmp_path / "test.txt").exists()

    async def test_write_allowed_when_user_approves(self, auto_ctx, tmp_path):
        with patch("forge.agent.permissions._prompt_user", new_callable=AsyncMock, return_value=True):
            result = await write_file(auto_ctx, "test.txt", "data")
            assert "Wrote" in result
            assert (tmp_path / "test.txt").read_text() == "data"

    async def test_edit_denied_when_user_says_no(self, auto_ctx, tmp_path):
        (tmp_path / "code.py").write_text("old_val = 1\n")
        with patch("forge.agent.permissions._prompt_user", new_callable=AsyncMock, return_value=False):
            result = await edit_file(auto_ctx, "code.py", "old_val", "new_val")
            assert "denied" in result.lower()
            assert "old_val" in (tmp_path / "code.py").read_text()

    async def test_command_denied(self, auto_ctx):
        with patch("forge.agent.permissions._prompt_user", new_callable=AsyncMock, return_value=False):
            result = await run_command(auto_ctx, "echo hello")
            assert "denied" in result.lower()


class TestAskPermissions:
    async def test_read_no_permission_check(self, ask_ctx, tmp_path):
        """read_file doesn't call check_permission, so it works even under ASK."""
        (tmp_path / "test.txt").write_text("content")
        result = await read_file(ask_ctx, "test.txt")
        assert "content" in result

    async def test_write_denied_under_ask(self, ask_ctx, tmp_path):
        with patch("forge.agent.permissions._prompt_user", new_callable=AsyncMock, return_value=False):
            result = await write_file(ask_ctx, "test.txt", "data")
            assert "denied" in result.lower()


class TestToolChaining:
    """Integration test: write → read → edit → verify."""

    async def test_write_read_edit_cycle(self, tmp_path):
        ctx = MagicMock()
        ctx.deps = AgentDeps(
            cwd=tmp_path,
            console=Console(file=None, force_terminal=False, no_color=True),
            permission=PermissionPolicy.YOLO,
        )

        # Write a file
        result = await write_file(ctx, "app.py", "name = 'old'\nversion = '1.0'\n")
        assert "Wrote" in result

        # Read it back
        result = await read_file(ctx, "app.py")
        assert "name = 'old'" in result

        # Edit it
        result = await edit_file(ctx, "app.py", "name = 'old'", "name = 'new'")
        assert "Edited" in result

        # Verify edit
        result = await read_file(ctx, "app.py")
        assert "name = 'new'" in result
        assert "version = '1.0'" in result
