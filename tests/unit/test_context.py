"""Tests for context management — history compaction and token estimation."""

import pytest

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)

from forge.agent.context import (
    estimate_tokens,
    compact_history,
    smart_compact_history,
    count_messages_tokens,
    _message_text,
    _message_to_readable,
    _truncate_tool_results,
    MIN_RECENT_MESSAGES,
    TOOL_RESULT_TRUNCATE,
)


def _make_user_msg(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _make_response(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _make_tool_result(tool_name: str, content: str, tool_call_id: str = "tc1") -> ModelRequest:
    return ModelRequest(parts=[ToolReturnPart(
        tool_name=tool_name,
        content=content,
        tool_call_id=tool_call_id,
    )])


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        assert estimate_tokens("hello world!") == 3

    def test_longer_string(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100


class TestMessageText:
    def test_user_message(self):
        msg = _make_user_msg("hello there")
        assert "hello there" in _message_text(msg)

    def test_response_message(self):
        msg = _make_response("model says hi")
        assert "model says hi" in _message_text(msg)

    def test_tool_result_message(self):
        msg = _make_tool_result("read_file", "file contents here")
        assert "file contents here" in _message_text(msg)


class TestTruncateToolResults:
    def test_short_content_unchanged(self):
        msg = _make_tool_result("read_file", "short content")
        result = _truncate_tool_results(msg)
        for part in result.parts:
            if isinstance(part, ToolReturnPart):
                assert "truncated" not in str(part.content)

    def test_long_content_truncated(self):
        long_content = "x" * 1000
        msg = _make_tool_result("read_file", long_content)
        result = _truncate_tool_results(msg, max_len=100)
        for part in result.parts:
            if isinstance(part, ToolReturnPart):
                assert "truncated" in str(part.content)
                assert len(str(part.content)) < len(long_content)

    def test_non_request_unchanged(self):
        msg = _make_response("hello")
        result = _truncate_tool_results(msg)
        assert result is msg


class TestCompactHistory:
    def test_empty_history(self):
        assert compact_history([]) == []

    def test_short_history_unchanged(self):
        msgs = [_make_user_msg("hi"), _make_response("hello")]
        result = compact_history(msgs)
        assert len(result) == 2

    def test_keeps_recent_messages(self):
        msgs = []
        for i in range(20):
            msgs.append(_make_user_msg(f"msg {i}"))
            msgs.append(_make_response(f"response {i}"))
        result = compact_history(msgs, token_budget=100)
        assert len(result) >= MIN_RECENT_MESSAGES

    def test_truncates_old_tool_results(self):
        long_tool_content = "x" * 2000
        msgs = [
            _make_tool_result("read_file", long_tool_content),
            _make_response("got it"),
            _make_user_msg("recent 1"),
            _make_response("recent 2"),
            _make_user_msg("recent 3"),
            _make_response("recent 4"),
        ]
        result = compact_history(msgs, token_budget=50000)
        if len(result) > MIN_RECENT_MESSAGES:
            first_msg = result[0]
            if isinstance(first_msg, ModelRequest):
                for part in first_msg.parts:
                    if isinstance(part, ToolReturnPart):
                        assert len(str(part.content)) < 2000

    def test_drops_oldest_when_over_budget(self):
        msgs = [_make_user_msg("x" * 1000) for _ in range(50)]
        result = compact_history(msgs, token_budget=500)
        assert len(result) < len(msgs)


class TestCountMessagesTokens:
    def test_count_empty(self):
        count, tokens = count_messages_tokens([])
        assert count == 0
        assert tokens == 0

    def test_count_messages(self):
        msgs = [_make_user_msg("hello"), _make_response("world")]
        count, tokens = count_messages_tokens(msgs)
        assert count == 2
        assert tokens > 0


class TestMessageToReadable:
    def test_user_message(self):
        msg = _make_user_msg("explain this code")
        result = _message_to_readable(msg)
        assert "User: explain this code" in result

    def test_response_message(self):
        msg = _make_response("here is the explanation")
        result = _message_to_readable(msg)
        assert "Assistant: here is the explanation" in result

    def test_tool_result_message(self):
        msg = _make_tool_result("read_file", "file contents here")
        result = _message_to_readable(msg)
        assert "Tool result (read_file)" in result
        assert "file contents here" in result

    def test_long_tool_result_truncated(self):
        msg = _make_tool_result("read_file", "x" * 500)
        result = _message_to_readable(msg)
        assert "..." in result
        assert len(result) < 500


class TestSmartCompactHistory:
    async def test_short_history_unchanged(self):
        msgs = [_make_user_msg("hi"), _make_response("hello")]
        result = await smart_compact_history(msgs)
        assert len(result) == 2

    async def test_falls_back_on_llm_failure(self):
        """When LLM is unavailable, falls back to mechanical compaction."""
        msgs = []
        for i in range(20):
            # Use longer messages so they exceed the token budget
            msgs.append(_make_user_msg(f"message number {i} " * 50))
            msgs.append(_make_response(f"response number {i} " * 50))

        # This will fail because no Ollama is running in tests,
        # falling back to mechanical compaction. Use small budget to force drops.
        result = await smart_compact_history(msgs, token_budget=500)
        assert len(result) < len(msgs)
        assert len(result) >= MIN_RECENT_MESSAGES
