"""Tool-call loop detection and circuit breaker.

Monitors tool calls via hooks and intervenes when the model is stuck in a loop.
Two-phase response: warn (inject redirect via ModelRetry) then trip (raise exception).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from forge.log import get_logger

if TYPE_CHECKING:
    from forge.agent.deps import AgentDeps

logger = get_logger(__name__)


@dataclass
class ToolCallRecord:
    tool_name: str
    args_hash: str
    succeeded: bool
    timestamp: float
    file_path: str = ""


@dataclass
class _State:
    warning_issued: bool = False
    tripped: bool = False
    trip_reason: str = ""
    post_warning_count: int = 0
    loop_start_index: int | None = None  # message index when looping was first detected


class CircuitBreakerTripped(Exception):
    """Raised when the circuit breaker trips after warning grace period."""


def _hash_args(args: dict) -> str:
    """Hash tool args for comparison. Truncate long values."""
    truncated = {}
    for k, v in sorted(args.items()):
        s = str(v)
        if len(s) > 1000:
            s = s[:1000]
        truncated[k] = s
    return hashlib.sha256(json.dumps(truncated, sort_keys=True).encode()).hexdigest()[:16]


# Benign repeat patterns: read after write on the same file is expected
_WRITE_TOOLS = frozenset({"edit_file", "write_file"})
_READ_TOOLS = frozenset({"read_file"})


class ToolCallTracker:
    """Monitors tool call patterns and detects loops."""

    def __init__(
        self,
        *,
        identical_threshold: int = 3,
        failure_threshold: int = 3,
        oscillation_window: int = 3,
        post_warning_grace: int = 2,
        history_size: int = 20,
    ) -> None:
        self._history: deque[ToolCallRecord] = deque(maxlen=history_size)
        self._state = _State()
        self._identical_threshold = identical_threshold
        self._failure_threshold = failure_threshold
        self._oscillation_window = oscillation_window
        self._post_warning_grace = post_warning_grace
        self._current_message_count: int = 0

    def record(self, tool_name: str, args: dict, succeeded: bool) -> None:
        """Record a tool call."""
        self._history.append(ToolCallRecord(
            tool_name=tool_name,
            args_hash=_hash_args(args),
            succeeded=succeeded,
            timestamp=time.monotonic(),
            file_path=args.get("file_path", ""),
        ))

    def check(self) -> str | None:
        """Check for loop patterns. Returns reason string or None."""
        if self._state.tripped:
            return self._state.trip_reason

        reason = (
            self._check_identical_repeat()
            or self._check_oscillation()
            or self._check_repeated_failures()
        )

        if reason and self._state.warning_issued:
            # Already warned — count towards trip (for non-hook code paths)
            self._state.post_warning_count += 1
            if self._state.post_warning_count >= self._post_warning_grace:
                self._state.tripped = True
                self._state.trip_reason = reason
        elif reason:
            self._state.warning_issued = True
            self._state.trip_reason = reason
            self._state.post_warning_count = 0
            # Record where in the message history looping started
            self._state.loop_start_index = self._current_message_count

        return reason

    def reset_state(self) -> None:
        """Reset warning/trip state at turn start. History persists."""
        self._state = _State()

    @property
    def warning_issued(self) -> bool:
        return self._state.warning_issued

    @property
    def tripped(self) -> bool:
        return self._state.tripped

    @property
    def trip_reason(self) -> str:
        return self._state.trip_reason

    @property
    def loop_start_index(self) -> int | None:
        """Message index when the loop pattern was first detected."""
        return self._state.loop_start_index

    def set_message_count(self, count: int) -> None:
        """Update the current message count (called from agent loop before each turn)."""
        self._current_message_count = count

    def _check_identical_repeat(self) -> str | None:
        """Same (name, args_hash) N times in a row."""
        if len(self._history) < self._identical_threshold:
            return None

        recent = list(self._history)[-self._identical_threshold:]
        first = recent[0]

        if not all(r.tool_name == first.tool_name and r.args_hash == first.args_hash for r in recent):
            return None

        if self._is_benign_repeat(recent):
            return None

        return f"called {first.tool_name} with identical arguments {self._identical_threshold} times in a row"

    def _check_oscillation(self) -> str | None:
        """Alternating A-B-A-B pattern with same args each time."""
        window = self._oscillation_window
        needed = window * 2  # full cycles

        if len(self._history) < needed:
            return None

        recent = list(self._history)[-needed:]

        # Check A-B alternating pattern
        a_name, a_hash = recent[0].tool_name, recent[0].args_hash
        b_name, b_hash = recent[1].tool_name, recent[1].args_hash

        if a_name == b_name:
            return None

        for i, r in enumerate(recent):
            if i % 2 == 0:
                if r.tool_name != a_name or r.args_hash != a_hash:
                    return None
            else:
                if r.tool_name != b_name or r.args_hash != b_hash:
                    return None

        return f"oscillating between {a_name} and {b_name} for {window} cycles"

    def _check_repeated_failures(self) -> str | None:
        """Same tool failing N times consecutively."""
        if len(self._history) < self._failure_threshold:
            return None

        recent = list(self._history)[-self._failure_threshold:]

        if not all(not r.succeeded for r in recent):
            return None

        first = recent[0]
        if not all(r.tool_name == first.tool_name for r in recent):
            return None

        return f"{first.tool_name} failed {self._failure_threshold} times in a row"

    def _is_benign_repeat(self, records: list[ToolCallRecord]) -> bool:
        """Check if the repeat pattern is benign (e.g. read after write)."""
        if len(records) < 2:
            return False

        # read_file after edit_file/write_file on the same file_path is OK
        current = records[-1]
        if current.tool_name in _READ_TOOLS and current.file_path:
            # Scan all history (not just until first non-read) for a write to the same path
            for prev in reversed(list(self._history)[:-1]):
                if prev.tool_name in _WRITE_TOOLS and prev.file_path == current.file_path:
                    return True

        return False


def _build_diagnostic(reason: str, deps: AgentDeps) -> str:
    """Build a context-aware diagnostic message for circuit breaker warnings.

    Includes concrete action suggestions and the specific tool/args to help the model
    understand exactly what it's repeating. This is the last message before trip.
    """
    parts = [f"⚠ LOOP DETECTED: {reason}. This tool call was BLOCKED (not executed)."]

    # Check if test failures exist — that's likely the root cause
    if deps.test_results:
        parts.append(
            "ACTION: Tests are failing — read the test error output in the system prompt "
            "and fix the ROOT CAUSE instead of retrying the same approach."
        )
    elif "identical" in reason:
        parts.append(
            "ACTION: Stop repeating this call. Instead: "
            "(1) Read the target file to check its current state, "
            "(2) use search_code to find related patterns, or "
            "(3) try a completely different strategy."
        )
    elif "oscillating" in reason:
        parts.append(
            "ACTION: You're going back and forth between two tools. Stop and reason: "
            "use <analysis> tags to think about what's actually wrong, "
            "then take ONE decisive action."
        )
    elif "failed" in reason:
        parts.append(
            "ACTION: This tool keeps failing. Read the error messages carefully. "
            "Common fixes: check file paths exist, verify command syntax, "
            "or try a fundamentally different approach."
        )
    else:
        parts.append("ACTION: Try a completely different approach, or ask the user for guidance.")

    parts.append("NEXT ATTEMPT WILL TRIP THE CIRCUIT BREAKER AND END THIS TURN.")

    return " ".join(parts)


def wire_circuit_breaker(
    tracker: ToolCallTracker,
    deps: AgentDeps,
) -> None:
    """Register circuit breaker hooks on deps.hook_registry."""
    from forge.agent.hooks import (
        HookAction,
        HookEscalation,
        HookResult,
        PostToolUse,
        PostToolUseFailure,
        PreToolUse,
        TurnStart,
    )

    registry = deps.hook_registry
    if registry is None:
        return

    async def _on_post_tool_use(event: PostToolUse) -> None:
        tracker.record(event.tool_name, event.args, succeeded=True)
        reason = tracker.check()
        if reason and tracker.warning_issued:
            logger.debug("Circuit breaker detected: %s", reason)
            # Signal escalator if available
            if deps.escalator:
                deps.escalator.add_signal("circuit_breaker", 3.0, 0)

    async def _on_post_tool_failure(event: PostToolUseFailure) -> None:
        tracker.record(event.tool_name, event.args, succeeded=False)
        reason = tracker.check()
        if reason and tracker.warning_issued:
            logger.debug("Circuit breaker detected (failure): %s", reason)
            if deps.escalator:
                deps.escalator.add_signal("circuit_breaker", 3.0, 0)

    # Make CircuitBreakerTripped bypass HookRegistry.check()'s except-Exception
    _TripEscalation = type(
        "CircuitBreakerTripped",
        (CircuitBreakerTripped, HookEscalation),
        {},
    )

    async def _on_pre_tool_use(event: PreToolUse) -> HookResult:
        if tracker.tripped:
            raise _TripEscalation(
                f"Circuit breaker tripped: {tracker.trip_reason}"
            )
        if tracker.warning_issued:
            # Check if the agent self-corrected by trying a different approach.
            # Speculatively add this call to history, run pattern checks, then remove.
            speculative = ToolCallRecord(
                tool_name=event.tool_name,
                args_hash=_hash_args(event.args),
                succeeded=True,
                timestamp=time.monotonic(),
            )
            tracker._history.append(speculative)
            still_looping = (
                tracker._check_identical_repeat()
                or tracker._check_oscillation()
                or tracker._check_repeated_failures()
            )
            tracker._history.pop()

            if not still_looping:
                # Agent broke the pattern — clear warning and allow
                logger.debug("Circuit breaker: agent self-corrected, clearing warning")
                tracker.reset_state()
                return HookResult()

            # Still looping — count toward trip
            tracker._state.post_warning_count += 1
            if tracker._state.post_warning_count >= tracker._post_warning_grace:
                tracker._state.tripped = True
                tracker._state.trip_reason = (
                    tracker._state.trip_reason or "continued looping after warning"
                )
                raise _TripEscalation(
                    f"Circuit breaker tripped: {tracker.trip_reason}"
                )
            # Build diagnostic message based on loop type
            reason = tracker.trip_reason or "repeated pattern detected"
            diagnostic = _build_diagnostic(reason, deps)
            return HookResult(
                action=HookAction.BLOCK,
                message=diagnostic,
            )
        return HookResult()

    async def _on_turn_start(event: TurnStart) -> None:
        tracker.reset_state()

    registry.on(PostToolUse, _on_post_tool_use)
    registry.on(PostToolUseFailure, _on_post_tool_failure)
    registry.on(PreToolUse, _on_pre_tool_use, priority=100)
    registry.on(TurnStart, _on_turn_start)
