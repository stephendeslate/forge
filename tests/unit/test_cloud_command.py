"""Unit tests for /cloud command handler."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from forge.agent.commands import cmd_cloud, CommandContext, CommandResult
from forge.agent.deps import AgentDeps
from forge.agent.permissions import PermissionPolicy


@pytest.fixture
def cloud_ctx(tmp_path):
    """CommandContext for testing /cloud."""
    console = MagicMock()
    deps = AgentDeps(
        cwd=tmp_path,
        console=console,
        permission=PermissionPolicy.YOLO,
        cloud_reasoning_enabled=False,
    )
    ctx = CommandContext(
        console=console,
        deps=deps,
        agent=MagicMock(),
        message_history=None,
        session_id="test-session",
        db=None,
        mcp_servers=[],
        rag_available=False,
        rag_project_name="",
        extra_tools=[],
        system="",
        turn_counter=0,
    )
    return ctx


class TestCmdCloud:
    """Tests for cmd_cloud handler."""

    @pytest.mark.asyncio
    async def test_toggle_on_with_env_key(self, cloud_ctx):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            result = await cmd_cloud(cloud_ctx, "")
        assert isinstance(result, CommandResult)
        assert cloud_ctx.deps.cloud_reasoning_enabled is True

    @pytest.mark.asyncio
    async def test_toggle_off(self, cloud_ctx):
        cloud_ctx.deps.cloud_reasoning_enabled = True
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            result = await cmd_cloud(cloud_ctx, "")
        assert cloud_ctx.deps.cloud_reasoning_enabled is False

    @pytest.mark.asyncio
    async def test_refuses_without_api_key(self, cloud_ctx):
        env = os.environ.copy()
        env.pop("GOOGLE_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            with patch("forge.agent.commands.settings") as mock_settings:
                mock_settings.gemini.api_key = ""
                result = await cmd_cloud(cloud_ctx, "")
        # Should stay off
        assert cloud_ctx.deps.cloud_reasoning_enabled is False
        # Should have printed error
        cloud_ctx.console.print.assert_called()
        call_args = str(cloud_ctx.console.print.call_args)
        assert "No API key" in call_args or "api_key" in call_args

    @pytest.mark.asyncio
    async def test_toggle_on_shows_warning(self, cloud_ctx):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            await cmd_cloud(cloud_ctx, "")
        call_args = str(cloud_ctx.console.print.call_args)
        assert "Google" in call_args or "cloud" in call_args.lower()
