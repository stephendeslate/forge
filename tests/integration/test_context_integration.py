"""Integration tests — context compaction mechanics (no LLM needed for tier 1)."""

import pytest
from unittest.mock import AsyncMock, patch

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from forge.agent.context import (
    TOOL_RESULT_TRUNCATE,
    _group_task_sequences,
    compact_history,
    count_messages_tokens,
    smart_compact_history,
)


def _user_msg(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _assistant_msg(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _tool_return_msg(name: str, content: str) -> ModelRequest:
    return ModelRequest(parts=[ToolReturnPart(
        tool_name=name,
        content=content,
        tool_call_id="test",
    )])


class TestCompactHistory:
    def test_empty_returns_empty(self):
        assert compact_history([]) == []

    def test_under_budget_unchanged(self):
        msgs = [_user_msg("hi"), _assistant_msg("hello")]
        result = compact_history(msgs, token_budget=100_000)
        assert len(result) == 2

    def test_truncates_old_tool_results(self):
        big_content = "x" * 5000
        msgs = [
            _tool_return_msg("read_file", big_content),
            _user_msg("a"),
            _assistant_msg("b"),
            _user_msg("c"),
            _assistant_msg("d"),
        ]
        result = compact_history(msgs, token_budget=100_000)
        # The old tool result should be truncated
        first = result[0]
        assert isinstance(first, ModelRequest)
        content = str(first.parts[0].content)
        assert len(content) <= TOOL_RESULT_TRUNCATE + 50  # +50 for "... (truncated)"

    def test_keeps_recent_messages(self):
        msgs = [_user_msg(f"msg-{i}") for i in range(10)]
        result = compact_history(msgs, token_budget=50)
        # Should always keep at least MIN_RECENT_MESSAGES (4)
        assert len(result) >= 4

    def test_drops_oldest_over_budget(self):
        msgs = [_user_msg("x" * 1000) for _ in range(20)]
        result = compact_history(msgs, token_budget=2000)
        assert len(result) < 20


class TestSmartCompactTier1:
    async def test_tier1_truncation_fits_budget(self):
        big = "x" * 10000
        msgs = [
            _tool_return_msg("read_file", big),
            _tool_return_msg("search_code", big),
            _user_msg("a"),
            _assistant_msg("b"),
            _user_msg("c"),
            _assistant_msg("d"),
            _user_msg("e"),
        ]
        # Large budget so tier 1 truncation is enough
        result = await smart_compact_history(msgs, token_budget=50_000)
        assert len(result) <= len(msgs)
        # Old tool results should be truncated
        for msg in result[:-4]:  # check older messages
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        assert len(str(part.content)) <= TOOL_RESULT_TRUNCATE + 50

    async def test_binary_content_stripped(self):
        from pydantic_ai.messages import BinaryContent

        msgs = [
            ModelRequest(parts=[
                BinaryContent(data=b"\x89PNG", media_type="image/png"),
            ]),
            _user_msg("a"),
            _assistant_msg("b"),
            _user_msg("c"),
            _assistant_msg("d"),
            _user_msg("e"),
        ]
        result = await smart_compact_history(msgs, token_budget=50_000)
        # Binary content should be replaced with text placeholder
        first = result[0]
        if isinstance(first, ModelRequest):
            for part in first.parts:
                assert not isinstance(part, BinaryContent)


class TestSmartCompactFallback:
    async def test_falls_back_on_llm_failure(self):
        msgs = [_user_msg("x" * 500) for _ in range(15)]
        with patch(
            "forge.agent.context._summarize_with_prompt",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await smart_compact_history(msgs, token_budget=500)
        # Should fall back to mechanical compaction
        assert len(result) < 15


class TestTaskSequenceGrouping:
    def test_groups_by_user_prompt(self):
        msgs = [
            _user_msg("first task"),
            _assistant_msg("on it"),
            _tool_return_msg("read_file", "content"),
            _user_msg("second task"),
            _assistant_msg("ok"),
        ]
        groups = _group_task_sequences(msgs)
        assert len(groups) == 2
        assert len(groups[0]) == 3
        assert len(groups[1]) == 2

    def test_single_group_no_prompt(self):
        msgs = [
            _assistant_msg("thinking"),
            _tool_return_msg("run_command", "output"),
        ]
        groups = _group_task_sequences(msgs)
        assert len(groups) == 1
