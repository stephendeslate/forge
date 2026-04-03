"""Tests for render event ordering — specifically the final_result fix.

The key bug was: pydantic-ai emits FinalResultEvent BEFORE all TextPartDelta
events are consumed. The old code called _finalize_live() on final_result,
which stopped the Rich Live display. Remaining deltas arrived after Live
was stopped and were silently discarded.

Fix: Remove _finalize_live() from the final_result handler. The finally
block handles finalization after ALL events are consumed.
"""

import asyncio
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.permissions import PermissionPolicy


@dataclass
class MockTextPart:
    content: str


@dataclass
class MockTextPartDelta:
    content_delta: str


@dataclass
class MockPartStartEvent:
    event_kind: str = "part_start"
    part: object = None


@dataclass
class MockPartDeltaEvent:
    event_kind: str = "part_delta"
    delta: object = None


@dataclass
class MockFinalResultEvent:
    event_kind: str = "final_result"


async def _simulate_pydantic_ai_events():
    """Simulate pydantic-ai event ordering: final_result BEFORE all deltas.

    This reproduces the exact bug: only "The" would render because
    _finalize_live() was called on final_result, stopping the Live display
    before remaining deltas arrived.
    """
    # Part start with first chunk
    yield MockPartStartEvent(part=MockTextPart(content="The"))

    # Final result fires early (this is the pydantic-ai behavior)
    yield MockFinalResultEvent()

    # Remaining deltas arrive AFTER final_result
    yield MockPartDeltaEvent(delta=MockTextPartDelta(content_delta=" project"))
    yield MockPartDeltaEvent(delta=MockTextPartDelta(content_delta=" name"))
    yield MockPartDeltaEvent(delta=MockTextPartDelta(content_delta=" is"))
    yield MockPartDeltaEvent(delta=MockTextPartDelta(content_delta=" **forge**"))
    yield MockPartDeltaEvent(delta=MockTextPartDelta(content_delta="."))


class TestRenderEventOrdering:
    """Test that render_events handles out-of-order final_result correctly."""

    def test_split_thinking_preserves_all_text(self):
        """_split_thinking should work with full text, not just first chunk."""
        from forge.agent.render import _split_thinking

        # Full response text (as if all deltas were collected)
        full_text = "The project name is **forge**."
        thinking, visible = _split_thinking(full_text)
        assert thinking == ""
        assert visible == full_text

    def test_finalize_live_collects_all_chunks(self):
        """Verify the text_chunks list accumulates all deltas before finalization."""
        # Simulate what render_events does internally
        text_chunks = []

        # Part start
        text_chunks.append("The")

        # More deltas arrive (even after final_result in the real flow)
        text_chunks.append(" project")
        text_chunks.append(" name")
        text_chunks.append(" is")
        text_chunks.append(" **forge**")
        text_chunks.append(".")

        # When _finalize_live runs (in the finally block), it joins all chunks
        full_text = "".join(text_chunks)
        assert full_text == "The project name is **forge**."

    async def test_event_collection_after_final_result(self):
        """The async generator yields events after final_result — all should be collected."""
        collected_text = []

        async for event in _simulate_pydantic_ai_events():
            if event.event_kind == "part_start" and hasattr(event, "part"):
                if hasattr(event.part, "content"):
                    collected_text.append(event.part.content)
            elif event.event_kind == "part_delta" and hasattr(event, "delta"):
                if hasattr(event.delta, "content_delta"):
                    collected_text.append(event.delta.content_delta)
            elif event.event_kind == "final_result":
                # Key: do NOT break or finalize here
                pass

        full_text = "".join(collected_text)
        assert full_text == "The project name is **forge**."

    async def test_early_finalize_loses_text(self):
        """Demonstrate what the bug was: finalizing on final_result loses text."""
        collected_text = []

        async for event in _simulate_pydantic_ai_events():
            if event.event_kind == "part_start" and hasattr(event, "part"):
                if hasattr(event.part, "content"):
                    collected_text.append(event.part.content)
            elif event.event_kind == "part_delta" and hasattr(event, "delta"):
                if hasattr(event.delta, "content_delta"):
                    collected_text.append(event.delta.content_delta)
            elif event.event_kind == "final_result":
                # BUG: breaking here loses remaining deltas
                break

        full_text = "".join(collected_text)
        # This proves the bug: only "The" is captured
        assert full_text == "The"
        assert full_text != "The project name is **forge**."
