"""Tests for circuit breaker loop detection."""

from __future__ import annotations

import pytest

from forge.agent.circuit_breaker import (
    CircuitBreakerTripped,
    ToolCallRecord,
    ToolCallTracker,
    _hash_args,
)


class TestHashArgs:
    def test_deterministic(self):
        h1 = _hash_args({"a": 1, "b": "hello"})
        h2 = _hash_args({"b": "hello", "a": 1})
        assert h1 == h2

    def test_truncates_long_values(self):
        h = _hash_args({"x": "a" * 5000})
        assert len(h) == 16


class TestIdenticalRepeat:
    def test_triggers_at_threshold(self):
        tracker = ToolCallTracker(identical_threshold=3)
        args = {"file_path": "/foo.py"}

        tracker.record("read_file", args, True)
        assert tracker.check() is None

        tracker.record("read_file", args, True)
        assert tracker.check() is None

        tracker.record("read_file", args, True)
        reason = tracker.check()
        assert reason is not None
        assert "identical" in reason
        assert tracker.warning_issued

    def test_below_threshold_no_trigger(self):
        tracker = ToolCallTracker(identical_threshold=3)
        args = {"file_path": "/foo.py"}

        tracker.record("read_file", args, True)
        tracker.record("read_file", args, True)
        assert tracker.check() is None
        assert not tracker.warning_issued

    def test_different_args_no_trigger(self):
        tracker = ToolCallTracker(identical_threshold=3)

        tracker.record("read_file", {"file_path": "/a.py"}, True)
        tracker.record("read_file", {"file_path": "/b.py"}, True)
        tracker.record("read_file", {"file_path": "/c.py"}, True)
        assert tracker.check() is None


class TestOscillation:
    def test_ab_pattern_triggers(self):
        tracker = ToolCallTracker(oscillation_window=3)
        a_args = {"file_path": "/a.py"}
        b_args = {"cmd": "ls"}

        # 3 full cycles = 6 calls: A B A B A B
        for _ in range(3):
            tracker.record("read_file", a_args, True)
            tracker.record("run_command", b_args, True)

        reason = tracker.check()
        assert reason is not None
        assert "oscillating" in reason

    def test_incomplete_oscillation_no_trigger(self):
        tracker = ToolCallTracker(oscillation_window=3)

        # Only 2 cycles
        for _ in range(2):
            tracker.record("read_file", {"file_path": "/a.py"}, True)
            tracker.record("run_command", {"cmd": "ls"}, True)

        assert tracker.check() is None


class TestRepeatedFailures:
    def test_consecutive_failures_trigger(self):
        # Use different args each time so identical repeat doesn't fire first
        tracker = ToolCallTracker(failure_threshold=3)

        tracker.record("run_command", {"cmd": "test1"}, False)
        tracker.record("run_command", {"cmd": "test2"}, False)
        tracker.record("run_command", {"cmd": "test3"}, False)

        reason = tracker.check()
        assert reason is not None
        assert "failed" in reason

    def test_mixed_success_no_trigger(self):
        tracker = ToolCallTracker(failure_threshold=3)

        tracker.record("run_command", {"cmd": "test1"}, False)
        tracker.record("run_command", {"cmd": "test2"}, True)  # success breaks streak
        tracker.record("run_command", {"cmd": "test3"}, False)

        assert tracker.check() is None

    def test_different_tools_no_trigger(self):
        tracker = ToolCallTracker(failure_threshold=3)

        tracker.record("run_command", {"cmd": "a"}, False)
        tracker.record("read_file", {"file_path": "/x"}, False)
        tracker.record("write_file", {"file_path": "/y"}, False)

        assert tracker.check() is None


class TestBenignRepeat:
    def test_read_after_write_exemption(self):
        tracker = ToolCallTracker(identical_threshold=3)

        tracker.record("edit_file", {"file_path": "/a.py"}, True)
        tracker.record("read_file", {"file_path": "/a.py"}, True)
        tracker.record("read_file", {"file_path": "/a.py"}, True)
        tracker.record("read_file", {"file_path": "/a.py"}, True)

        # read_file after edit_file should be exempted
        assert tracker.check() is None


class TestTwoPhaseResponse:
    def test_warning_then_trip_grace_1(self):
        """Default grace=1: warning → one more attempt → trip."""
        tracker = ToolCallTracker(identical_threshold=3, post_warning_grace=1)
        args = {"file_path": "/foo.py"}

        # 3 identical → warning
        for _ in range(3):
            tracker.record("read_file", args, True)
        tracker.check()
        assert tracker.warning_issued
        assert not tracker.tripped

        # 1 more → trip (grace=1)
        tracker.record("read_file", args, True)
        tracker.check()
        assert tracker.tripped
        assert tracker.trip_reason

    def test_warning_then_trip_grace_2(self):
        """Grace=2 still works for users who configure it."""
        tracker = ToolCallTracker(identical_threshold=3, post_warning_grace=2)
        args = {"file_path": "/foo.py"}

        # 3 identical → warning
        for _ in range(3):
            tracker.record("read_file", args, True)
        tracker.check()
        assert tracker.warning_issued
        assert not tracker.tripped

        # 1 more — not yet tripped
        tracker.record("read_file", args, True)
        tracker.check()
        assert not tracker.tripped

        # 2nd after warning → trip
        tracker.record("read_file", args, True)
        tracker.check()
        assert tracker.tripped
        assert tracker.trip_reason


class TestStateReset:
    def test_reset_clears_warning_keeps_history(self):
        tracker = ToolCallTracker(identical_threshold=3)
        args = {"file_path": "/foo.py"}

        for _ in range(3):
            tracker.record("read_file", args, True)
        tracker.check()
        assert tracker.warning_issued

        tracker.reset_state()
        assert not tracker.warning_issued
        assert not tracker.tripped
        assert len(tracker._history) > 0  # history persists


class TestLoopStartIndex:
    def test_records_index_on_first_warning(self):
        tracker = ToolCallTracker(identical_threshold=3)
        tracker.set_message_count(10)
        args = {"file_path": "/foo.py"}

        for _ in range(3):
            tracker.record("read_file", args, True)
        tracker.check()
        assert tracker.warning_issued
        assert tracker.loop_start_index == 10

    def test_no_index_before_warning(self):
        tracker = ToolCallTracker(identical_threshold=3)
        tracker.set_message_count(5)
        args = {"file_path": "/foo.py"}

        tracker.record("read_file", args, True)
        tracker.record("read_file", args, True)
        tracker.check()
        assert tracker.loop_start_index is None

    def test_reset_clears_index(self):
        tracker = ToolCallTracker(identical_threshold=3)
        tracker.set_message_count(10)
        args = {"file_path": "/foo.py"}

        for _ in range(3):
            tracker.record("read_file", args, True)
        tracker.check()
        assert tracker.loop_start_index == 10

        tracker.reset_state()
        assert tracker.loop_start_index is None


class TestSelfCorrectionDetection:
    """Verify that pattern checks work for speculative recheck (used by hook)."""

    def test_different_args_breaks_identical_pattern(self):
        """After identical loop detected, a call with different args breaks the pattern."""
        tracker = ToolCallTracker(identical_threshold=3, post_warning_grace=2)
        args_a = {"file_path": "/a.py"}
        args_b = {"file_path": "/b.py"}

        for _ in range(3):
            tracker.record("read_file", args_a, True)
        tracker.check()
        assert tracker.warning_issued

        # Speculative check: would a different call continue the loop?
        from forge.agent.circuit_breaker import ToolCallRecord, _hash_args
        import time

        speculative = ToolCallRecord("read_file", _hash_args(args_b), True, time.monotonic())
        tracker._history.append(speculative)
        still_looping = tracker._check_identical_repeat()
        tracker._history.pop()

        assert still_looping is None, "Different args should break identical pattern"

    def test_same_args_continues_identical_pattern(self):
        """After identical loop detected, same call continues the pattern."""
        tracker = ToolCallTracker(identical_threshold=3, post_warning_grace=2)
        args = {"file_path": "/a.py"}

        for _ in range(3):
            tracker.record("read_file", args, True)
        tracker.check()
        assert tracker.warning_issued

        from forge.agent.circuit_breaker import ToolCallRecord, _hash_args
        import time

        speculative = ToolCallRecord("read_file", _hash_args(args), True, time.monotonic())
        tracker._history.append(speculative)
        still_looping = tracker._check_identical_repeat()
        tracker._history.pop()

        assert still_looping is not None, "Same args should continue identical pattern"

    def test_different_tool_breaks_failure_pattern(self):
        """After repeated failures, calling a different tool breaks the pattern."""
        tracker = ToolCallTracker(failure_threshold=3, post_warning_grace=2)

        for _ in range(3):
            tracker.record("run_command", {"cmd": "pytest"}, False)
        tracker.check()
        assert tracker.warning_issued

        from forge.agent.circuit_breaker import ToolCallRecord, _hash_args
        import time

        speculative = ToolCallRecord("read_file", _hash_args({"file_path": "/err.log"}), True, time.monotonic())
        tracker._history.append(speculative)
        still_looping = tracker._check_repeated_failures()
        tracker._history.pop()

        assert still_looping is None, "Different tool should break failure pattern"


class TestDefaultGrace:
    def test_default_grace_is_2(self):
        """Default grace=2 ensures the diagnostic BLOCK is reachable."""
        tracker = ToolCallTracker(identical_threshold=3)
        assert tracker._post_warning_grace == 2


class TestMixedVariedCalls:
    def test_no_false_positive(self):
        tracker = ToolCallTracker()

        # Simulate typical varied agent work
        tracker.record("read_file", {"file_path": "/a.py"}, True)
        tracker.record("search_code", {"pattern": "def foo"}, True)
        tracker.record("read_file", {"file_path": "/b.py"}, True)
        tracker.record("edit_file", {"file_path": "/a.py", "old_string": "x", "new_string": "y"}, True)
        tracker.record("read_file", {"file_path": "/a.py"}, True)
        tracker.record("run_command", {"cmd": "pytest"}, True)

        assert tracker.check() is None
        assert not tracker.warning_issued
