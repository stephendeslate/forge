"""Tests for conversation forking on Gemini recovery."""

from __future__ import annotations

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from forge.agent.circuit_breaker import ToolCallTracker
from forge.agent.loop import _fork_history


def _make_messages(n: int) -> list:
    """Create n pairs of (request, response) messages."""
    msgs = []
    for i in range(n):
        msgs.append(ModelRequest(parts=[UserPromptPart(content=f"user msg {i}")]))
        msgs.append(ModelResponse(parts=[TextPart(content=f"assistant msg {i}")]))
    return msgs


class TestForkHistory:
    def test_truncates_at_loop_start_index(self):
        msgs = _make_messages(10)  # 20 messages total
        tracker = ToolCallTracker(identical_threshold=3)
        tracker._current_message_count = 14
        # Simulate warning being issued at message index 14
        tracker._state.loop_start_index = 14

        result = _fork_history(msgs, tracker)
        # Should keep first 14 messages + 1 bridge
        assert len(result) == 15
        # Last message should be the synthetic bridge
        bridge = result[-1]
        assert isinstance(bridge, ModelRequest)
        assert "loop" in bridge.parts[0].content.lower()

    def test_fallback_without_loop_start(self):
        msgs = _make_messages(10)  # 20 messages
        tracker = ToolCallTracker(identical_threshold=3)
        # No loop_start_index set

        result = _fork_history(msgs, tracker, identical_threshold=3)
        # Fallback removes 2 * 3 = 6 messages from the end
        assert len(result) == 20 - 6 + 1  # 14 kept + 1 bridge = 15

    def test_no_circuit_breaker(self):
        msgs = _make_messages(5)  # 10 messages
        result = _fork_history(msgs, None, identical_threshold=3)
        # Fallback: remove 6 from 10 → 4 + 1 bridge = 5
        assert len(result) == 5

    def test_empty_history(self):
        result = _fork_history([], None)
        assert result == []

    def test_short_history_preserves_first(self):
        msgs = _make_messages(2)  # 4 messages
        result = _fork_history(msgs, None, identical_threshold=3)
        # 2*3=6 > 4, so fallback keeps first message + bridge
        assert len(result) == 2
        assert isinstance(result[-1], ModelRequest)
        assert "loop" in result[-1].parts[0].content.lower()

    def test_bridge_message_content(self):
        msgs = _make_messages(10)
        tracker = ToolCallTracker(identical_threshold=3)
        tracker._state.loop_start_index = 16

        result = _fork_history(msgs, tracker)
        bridge = result[-1]
        assert "cloud model" in bridge.parts[0].content.lower()
        assert "stuck" in bridge.parts[0].content.lower()
