"""Tests for the hooks system."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.hooks import (
    HookAction,
    HookRegistry,
    HookResult,
    PostToolUse,
    PostToolUseFailure,
    PreToolUse,
    SessionStart,
    TurnEnd,
    TurnStart,
    permission_hook,
    with_hooks,
)
from forge.agent.permissions import PermissionPolicy


@pytest.fixture
def registry():
    return HookRegistry()


@pytest.fixture
def deps(tmp_path):
    return AgentDeps(
        cwd=tmp_path,
        console=Console(file=None, force_terminal=False, no_color=True),
        permission=PermissionPolicy.YOLO,
    )


class TestHookRegistry:
    async def test_emit_fires_all_handlers(self, registry):
        calls = []

        async def h1(event):
            calls.append("h1")

        async def h2(event):
            calls.append("h2")

        registry.on(SessionStart, h1)
        registry.on(SessionStart, h2)

        await registry.emit(SessionStart(session_id="s1", cwd="/tmp", permission="yolo"))
        assert sorted(calls) == ["h1", "h2"]

    async def test_emit_no_handlers(self, registry):
        # Should not raise
        await registry.emit(SessionStart(session_id="s1", cwd="/tmp", permission="yolo"))

    async def test_check_returns_first_non_allow(self, registry):
        async def blocker(event):
            return HookResult(action=HookAction.BLOCK, message="nope")

        async def allower(event):
            return HookResult()

        registry.on(PreToolUse, allower, priority=0)
        registry.on(PreToolUse, blocker, priority=1)

        result = await registry.check(PreToolUse(tool_name="write_file", args={}))
        assert result.action == HookAction.BLOCK
        assert result.message == "nope"

    async def test_check_all_allow(self, registry):
        async def allower(event):
            return HookResult()

        registry.on(PreToolUse, allower)
        result = await registry.check(PreToolUse(tool_name="read_file", args={}))
        assert result.action == HookAction.ALLOW

    async def test_check_no_handlers(self, registry):
        result = await registry.check(PreToolUse(tool_name="read_file", args={}))
        assert result.action == HookAction.ALLOW

    async def test_priority_ordering(self, registry):
        calls = []

        async def h_low(event):
            calls.append("low")
            return HookResult()

        async def h_high(event):
            calls.append("high")
            return HookResult()

        registry.on(PreToolUse, h_high, priority=10)
        registry.on(PreToolUse, h_low, priority=1)

        await registry.check(PreToolUse(tool_name="read_file", args={}))
        assert calls == ["low", "high"]

    async def test_sync_handler(self, registry):
        calls = []

        def sync_handler(event):
            calls.append("sync")

        registry.on(SessionStart, sync_handler)
        await registry.emit(SessionStart(session_id="s1", cwd="/tmp", permission="yolo"))
        assert calls == ["sync"]


class TestWithHooks:
    async def test_passthrough_without_registry(self, deps):
        """When hook_registry is None, function passes through unchanged."""
        async def my_tool(ctx, file_path: str) -> str:
            return f"read {file_path}"

        wrapped = with_hooks(my_tool)
        ctx = MagicMock()
        ctx.deps = deps
        assert deps.hook_registry is None

        result = await wrapped(ctx, file_path="test.txt")
        assert result == "read test.txt"

    async def test_hooks_fire_on_success(self, deps):
        """PreToolUse and PostToolUse fire when registry exists."""
        registry = HookRegistry()
        deps.hook_registry = registry
        events = []

        async def track_pre(event):
            events.append(("pre", event.tool_name))
            return HookResult()

        async def track_post(event):
            events.append(("post", event.tool_name, event.result))

        registry.on(PreToolUse, track_pre)
        registry.on(PostToolUse, track_post)

        async def my_tool(ctx, file_path: str) -> str:
            return f"read {file_path}"

        wrapped = with_hooks(my_tool)
        ctx = MagicMock()
        ctx.deps = deps

        result = await wrapped(ctx, file_path="test.txt")
        assert result == "read test.txt"
        assert ("pre", "my_tool") in events
        assert ("post", "my_tool", "read test.txt") in events

    async def test_block_raises_model_retry(self, deps):
        """When PreToolUse handler returns BLOCK, ModelRetry is raised."""
        from pydantic_ai import ModelRetry

        registry = HookRegistry()
        deps.hook_registry = registry

        async def blocker(event):
            return HookResult(action=HookAction.BLOCK, message="blocked!")

        registry.on(PreToolUse, blocker)

        async def my_tool(ctx, file_path: str) -> str:
            return "should not reach"

        wrapped = with_hooks(my_tool)
        ctx = MagicMock()
        ctx.deps = deps

        with pytest.raises(ModelRetry, match="blocked!"):
            await wrapped(ctx, file_path="test.txt")

    async def test_failure_emits_post_tool_use_failure(self, deps):
        """When tool raises, PostToolUseFailure fires."""
        registry = HookRegistry()
        deps.hook_registry = registry
        failures = []

        async def track_failure(event):
            failures.append(event.error)

        registry.on(PostToolUseFailure, track_failure)

        async def failing_tool(ctx) -> str:
            raise ValueError("boom")

        wrapped = with_hooks(failing_tool)
        ctx = MagicMock()
        ctx.deps = deps

        with pytest.raises(ValueError, match="boom"):
            await wrapped(ctx)

        assert len(failures) == 1
        assert str(failures[0]) == "boom"


class TestPermissionHook:
    async def test_yolo_allows_everything(self, deps):
        deps.permission = PermissionPolicy.YOLO
        event = PreToolUse(tool_name="write_file", args={"file_path": "x"})
        result = await permission_hook(event, deps)
        assert result.action == HookAction.ALLOW

    async def test_auto_allows_safe_tools(self, deps):
        deps.permission = PermissionPolicy.AUTO
        event = PreToolUse(tool_name="read_file", args={"file_path": "x"})
        result = await permission_hook(event, deps)
        assert result.action == HookAction.ALLOW

    async def test_auto_blocks_dangerous_when_denied(self, deps):
        deps.permission = PermissionPolicy.AUTO
        event = PreToolUse(tool_name="write_file", args={"file_path": "x"})

        from unittest.mock import patch

        with patch("forge.agent.permissions._prompt_user", new_callable=AsyncMock, return_value=False):
            result = await permission_hook(event, deps)
        assert result.action == HookAction.BLOCK

    async def test_auto_allows_dangerous_when_approved(self, deps):
        deps.permission = PermissionPolicy.AUTO
        event = PreToolUse(tool_name="run_command", args={"command": "ls"})

        from unittest.mock import patch

        with patch("forge.agent.permissions._prompt_user", new_callable=AsyncMock, return_value=True):
            result = await permission_hook(event, deps)
        assert result.action == HookAction.ALLOW
