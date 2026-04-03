"""Integration tests — circuit breaker wired through real hooks."""

import pytest
from unittest.mock import MagicMock

from pydantic_ai import ModelRetry
from rich.console import Console

from forge.agent.circuit_breaker import CircuitBreakerTripped, ToolCallTracker, wire_circuit_breaker
from forge.agent.deps import AgentDeps
from forge.agent.hooks import (
    HookEscalation,
    HookRegistry,
    PostToolUse,
    PreToolUse,
    TurnStart,
    make_permission_handler,
    with_hooks,
)
from forge.agent.permissions import PermissionPolicy
from forge.agent.tools import read_file, write_file

_read = with_hooks(read_file)
_write = with_hooks(write_file)


@pytest.fixture
def cb_deps(tmp_cwd):
    """Deps with circuit breaker wired: identical=3, grace=2."""
    registry = HookRegistry()
    console = Console(file=None, force_terminal=False, no_color=True)
    tracker = ToolCallTracker(
        identical_threshold=3,
        post_warning_grace=2,
    )
    deps = AgentDeps(
        cwd=tmp_cwd,
        console=console,
        permission=PermissionPolicy.YOLO,
        hook_registry=registry,
        circuit_breaker=tracker,
    )
    registry.on(PreToolUse, make_permission_handler(deps))
    wire_circuit_breaker(tracker, deps)
    return deps, tracker


@pytest.fixture
def cb_ctx(cb_deps):
    deps, _ = cb_deps
    ctx = MagicMock()
    ctx.deps = deps
    return ctx


class TestCircuitBreakerWired:
    async def test_normal_calls_not_blocked(self, cb_ctx, tmp_cwd):
        """Two identical calls (below threshold) succeed."""
        (tmp_cwd / "a.txt").write_text("hello")
        r1 = await _read(cb_ctx, "a.txt")
        r2 = await _read(cb_ctx, "a.txt")
        assert "hello" in r1
        assert "hello" in r2

    async def test_identical_calls_trigger_warning(self, cb_ctx, tmp_cwd):
        """3 identical calls trigger warning → PreToolUse blocks with loop message."""
        (tmp_cwd / "a.txt").write_text("hello")
        await _read(cb_ctx, "a.txt")
        await _read(cb_ctx, "a.txt")
        await _read(cb_ctx, "a.txt")
        # 4th call: tracker has warned, PreToolUse should BLOCK
        with pytest.raises(ModelRetry, match="loop"):
            await _read(cb_ctx, "a.txt")

    async def test_trip_after_grace(self, cb_ctx, tmp_cwd):
        """After warning + grace exhausted → CircuitBreakerTripped."""
        (tmp_cwd / "a.txt").write_text("hello")
        await _read(cb_ctx, "a.txt")
        await _read(cb_ctx, "a.txt")
        await _read(cb_ctx, "a.txt")

        # 4th call: warning issued, BLOCK (grace count 1 of 2)
        with pytest.raises(ModelRetry):
            await _read(cb_ctx, "a.txt")

        # 5th call: grace count 2 of 2 → trip
        with pytest.raises((CircuitBreakerTripped, HookEscalation)):
            await _read(cb_ctx, "a.txt")

    async def test_turn_reset_clears_state(self, cb_deps, cb_ctx, tmp_cwd):
        """TurnStart resets warning state so calls succeed again."""
        deps, tracker = cb_deps
        registry = deps.hook_registry

        (tmp_cwd / "a.txt").write_text("hello")
        await _read(cb_ctx, "a.txt")
        await _read(cb_ctx, "a.txt")
        await _read(cb_ctx, "a.txt")

        # Warning issued — emit TurnStart to reset
        await registry.emit(TurnStart(turn_number=2, prompt="next"))

        # Should succeed again
        result = await _read(cb_ctx, "a.txt")
        assert "hello" in result

    async def test_different_calls_no_trigger(self, cb_ctx, tmp_cwd):
        """Different tool calls don't trigger the circuit breaker."""
        (tmp_cwd / "a.txt").write_text("aaa")
        (tmp_cwd / "b.txt").write_text("bbb")
        (tmp_cwd / "c.txt").write_text("ccc")

        r1 = await _read(cb_ctx, "a.txt")
        r2 = await _read(cb_ctx, "b.txt")
        r3 = await _read(cb_ctx, "c.txt")
        assert "aaa" in r1
        assert "bbb" in r2
        assert "ccc" in r3
