"""Tests for agent session resume — TypeAdapter round-trip and history loading."""

import pytest

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from forge.agent.loop import _message_list_adapter


def _make_user_msg(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _make_response(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


class TestTypeAdapterRoundTrip:
    """Verify TypeAdapter serializes and deserializes ModelMessage lists."""

    def test_simple_round_trip(self):
        messages: list[ModelMessage] = [
            _make_user_msg("hello"),
            _make_response("world"),
        ]
        json_bytes = _message_list_adapter.dump_json(messages)
        restored = _message_list_adapter.validate_json(json_bytes)
        assert len(restored) == 2
        assert isinstance(restored[0], ModelRequest)
        assert isinstance(restored[1], ModelResponse)

    def test_empty_list(self):
        json_bytes = _message_list_adapter.dump_json([])
        restored = _message_list_adapter.validate_json(json_bytes)
        assert restored == []

    def test_multi_turn_round_trip(self):
        messages: list[ModelMessage] = []
        for i in range(5):
            messages.append(_make_user_msg(f"question {i}"))
            messages.append(_make_response(f"answer {i}"))

        json_bytes = _message_list_adapter.dump_json(messages)
        restored = _message_list_adapter.validate_json(json_bytes)
        assert len(restored) == 10

        # Verify content is preserved
        first_req = restored[0]
        assert isinstance(first_req, ModelRequest)
        assert any(
            hasattr(p, "content") and "question 0" in str(p.content)
            for p in first_req.parts
        )

    def test_invalid_json_returns_error(self):
        with pytest.raises(Exception):
            _message_list_adapter.validate_json(b"not valid json")
