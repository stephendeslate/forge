"""Tests for hook feedback collection (emit_and_collect_feedback)."""

import pytest
from unittest.mock import MagicMock

from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.hooks import (
    HookAction,
    HookRegistry,
    HookResult,
    PostToolUse,
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


class TestEmitAndCollectFeedback:
    async def test_collects_feedback_strings(self, registry):
        async def handler_with_feedback(event):
            return HookResult(feedback="lint: 2 warnings")

        registry.on(PostToolUse, handler_with_feedback)

        event = PostToolUse(tool_name="write_file", args={}, result="ok", elapsed=0.1)
        feedbacks = await registry.emit_and_collect_feedback(event)
        assert feedbacks == ["lint: 2 warnings"]

    async def test_multiple_feedbacks(self, registry):
        async def handler1(event):
            return HookResult(feedback="feedback 1")

        async def handler2(event):
            return HookResult(feedback="feedback 2")

        registry.on(PostToolUse, handler1)
        registry.on(PostToolUse, handler2)

        event = PostToolUse(tool_name="write_file", args={}, result="ok", elapsed=0.1)
        feedbacks = await registry.emit_and_collect_feedback(event)
        assert len(feedbacks) == 2
        assert "feedback 1" in feedbacks
        assert "feedback 2" in feedbacks

    async def test_no_feedback_returns_empty(self, registry):
        async def handler_no_feedback(event):
            return HookResult()  # No feedback field

        registry.on(PostToolUse, handler_no_feedback)

        event = PostToolUse(tool_name="read_file", args={}, result="ok", elapsed=0.1)
        feedbacks = await registry.emit_and_collect_feedback(event)
        assert feedbacks == []

    async def test_none_handler_returns_empty(self, registry):
        async def handler_returns_none(event):
            return None

        registry.on(PostToolUse, handler_returns_none)

        event = PostToolUse(tool_name="read_file", args={}, result="ok", elapsed=0.1)
        feedbacks = await registry.emit_and_collect_feedback(event)
        assert feedbacks == []

    async def test_error_handler_skipped(self, registry):
        async def broken_handler(event):
            raise RuntimeError("handler broke")

        async def good_handler(event):
            return HookResult(feedback="still works")

        registry.on(PostToolUse, broken_handler)
        registry.on(PostToolUse, good_handler)

        event = PostToolUse(tool_name="read_file", args={}, result="ok", elapsed=0.1)
        feedbacks = await registry.emit_and_collect_feedback(event)
        assert feedbacks == ["still works"]

    async def test_no_handlers(self, registry):
        event = PostToolUse(tool_name="read_file", args={}, result="ok", elapsed=0.1)
        feedbacks = await registry.emit_and_collect_feedback(event)
        assert feedbacks == []


class TestWithHooksFeedbackMerge:
    async def test_feedback_appended_to_result(self, deps):
        """PostToolUse feedback gets appended to the string result."""
        registry = HookRegistry()
        deps.hook_registry = registry

        async def lint_hook(event):
            return HookResult(feedback="[lint] 1 warning in file.py")

        from forge.agent.hooks import PreToolUse
        registry.on(PostToolUse, lint_hook)

        async def my_tool(ctx, file_path: str) -> str:
            return "File written successfully"

        wrapped = with_hooks(my_tool)
        ctx = MagicMock()
        ctx.deps = deps

        result = await wrapped(ctx, file_path="file.py")
        assert "File written successfully" in result
        assert "[lint] 1 warning" in result

    async def test_legacy_feedback_also_merged(self, deps):
        """deps._post_tool_feedback is also merged into the result."""
        registry = HookRegistry()
        deps.hook_registry = registry
        deps._post_tool_feedback = ["syntax check: ok"]

        async def my_tool(ctx) -> str:
            return "done"

        wrapped = with_hooks(my_tool)
        ctx = MagicMock()
        ctx.deps = deps

        result = await wrapped(ctx)
        assert "done" in result
        assert "syntax check: ok" in result
        # Legacy feedback should be cleared after use
        assert deps._post_tool_feedback == []

    async def test_no_feedback_result_unchanged(self, deps):
        """When no feedback, result stays as-is."""
        registry = HookRegistry()
        deps.hook_registry = registry

        async def my_tool(ctx) -> str:
            return "clean result"

        wrapped = with_hooks(my_tool)
        ctx = MagicMock()
        ctx.deps = deps

        result = await wrapped(ctx)
        assert result == "clean result"
