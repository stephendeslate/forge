"""Integration tests — agent tools through the with_hooks wrapper.

Key difference from unit tests: these call with_hooks(tool) so that
PreToolUse/PostToolUse hooks actually fire.  When a hook BLOCKs,
with_hooks raises ``pydantic_ai.ModelRetry`` (not a string return).
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from pydantic_ai import ModelRetry
from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.hooks import HookRegistry, PreToolUse, make_permission_handler, with_hooks
from forge.agent.permissions import PermissionPolicy
from forge.agent.tools import read_file, write_file, edit_file, run_command

# Module-level wrapped functions — these go through hooks
_read = with_hooks(read_file)
_write = with_hooks(write_file)
_edit = with_hooks(edit_file)
_run = with_hooks(run_command)


class TestAutoPermissions:
    """AUTO policy: reads allowed, writes/commands prompt the user."""

    async def test_read_allowed_without_prompt(self, hooked_ctx, tmp_cwd):
        (tmp_cwd / "test.txt").write_text("content")
        result = await _read(hooked_ctx, "test.txt")
        assert "content" in result

    async def test_write_blocked_when_user_denies(self, hooked_ctx, tmp_cwd):
        with patch(
            "forge.agent.permissions._prompt_user",
            new_callable=AsyncMock,
            return_value=False,
        ):
            with pytest.raises(ModelRetry, match="denied"):
                await _write(hooked_ctx, "test.txt", "data")
            assert not (tmp_cwd / "test.txt").exists()

    async def test_write_allowed_when_user_approves(self, hooked_ctx, tmp_cwd):
        with patch(
            "forge.agent.permissions._prompt_user",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await _write(hooked_ctx, "test.txt", "data")
            assert "Wrote" in result
            assert (tmp_cwd / "test.txt").read_text() == "data"

    async def test_edit_blocked_when_user_denies(self, hooked_ctx, tmp_cwd):
        (tmp_cwd / "code.py").write_text("old_val = 1\n")
        with patch(
            "forge.agent.permissions._prompt_user",
            new_callable=AsyncMock,
            return_value=False,
        ):
            with pytest.raises(ModelRetry, match="denied"):
                await _edit(hooked_ctx, "code.py", "old_val", "new_val")
            assert "old_val" in (tmp_cwd / "code.py").read_text()

    async def test_command_blocked_when_user_denies(self, hooked_ctx):
        with patch(
            "forge.agent.permissions._prompt_user",
            new_callable=AsyncMock,
            return_value=False,
        ):
            with pytest.raises(ModelRetry, match="denied"):
                await _run(hooked_ctx, "echo hello")


class TestYoloPermissions:
    """YOLO policy: everything allowed, no prompts."""

    async def test_write_always_allowed(self, tmp_cwd, console):
        registry = HookRegistry()
        deps = AgentDeps(
            cwd=tmp_cwd,
            console=console,
            permission=PermissionPolicy.YOLO,
            hook_registry=registry,
        )
        registry.on(PreToolUse, make_permission_handler(deps))
        ctx = MagicMock()
        ctx.deps = deps

        result = await _write(ctx, "test.txt", "yolo data")
        assert "Wrote" in result
        assert (tmp_cwd / "test.txt").read_text() == "yolo data"


class TestToolChaining:
    """Integration test: write → read → edit → verify through hooks."""

    async def test_write_read_edit_cycle(self, tmp_cwd, console):
        registry = HookRegistry()
        deps = AgentDeps(
            cwd=tmp_cwd,
            console=console,
            permission=PermissionPolicy.YOLO,
            hook_registry=registry,
        )
        registry.on(PreToolUse, make_permission_handler(deps))
        ctx = MagicMock()
        ctx.deps = deps

        # Write
        result = await _write(ctx, "app.py", "name = 'old'\nversion = '1.0'\n")
        assert "Wrote" in result

        # Read
        result = await _read(ctx, "app.py")
        assert "name = 'old'" in result

        # Edit
        result = await _edit(ctx, "app.py", "name = 'old'", "name = 'new'")
        assert "Edited" in result

        # Verify
        result = await _read(ctx, "app.py")
        assert "name = 'new'" in result
        assert "version = '1.0'" in result
