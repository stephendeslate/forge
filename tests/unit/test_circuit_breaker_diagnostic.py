"""Tests for circuit breaker diagnostic messages (Feature 2 enhancement)."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from forge.agent.circuit_breaker import _build_diagnostic
from forge.agent.deps import AgentDeps
from forge.agent.permissions import PermissionPolicy


@pytest.fixture
def deps(tmp_path):
    return AgentDeps(
        cwd=tmp_path,
        console=Console(file=None, force_terminal=False, no_color=True),
        permission=PermissionPolicy.YOLO,
    )


class TestBuildDiagnostic:
    def test_with_test_failures(self, deps):
        deps.test_results = "FAILED test_foo.py::test_bar"
        msg = _build_diagnostic("identical calls", deps)
        assert "LOOP DETECTED" in msg
        assert "Tests are failing" in msg

    def test_identical_calls(self, deps):
        msg = _build_diagnostic("called X with identical arguments 3 times", deps)
        assert "Stop repeating" in msg

    def test_oscillating(self, deps):
        msg = _build_diagnostic("oscillating between edit_file and read_file", deps)
        assert "back and forth" in msg

    def test_repeated_failures(self, deps):
        msg = _build_diagnostic("run_command failed 3 times in a row", deps)
        assert "keeps failing" in msg

    def test_generic_reason(self, deps):
        msg = _build_diagnostic("unknown pattern", deps)
        assert "completely different approach" in msg

    def test_test_failures_take_priority(self, deps):
        """Test failures should be mentioned even for oscillation reasons."""
        deps.test_results = "FAILED"
        msg = _build_diagnostic("oscillating between X and Y", deps)
        assert "Tests are failing" in msg
        assert "back and forth" not in msg

    def test_includes_trip_warning(self, deps):
        """All diagnostics should warn about imminent trip."""
        msg = _build_diagnostic("identical calls", deps)
        assert "NEXT ATTEMPT WILL TRIP" in msg
