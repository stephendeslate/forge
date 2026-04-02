"""Tests for model escalation."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.escalation import ModelEscalator


def _make_settings():
    mock = MagicMock()
    mock.ollama.fast_model = "qwen3.5:4b"
    mock.ollama.heavy_model = "qwen3-coder-next:q8_0"
    return mock


@pytest.fixture
def console():
    return Console(file=io.StringIO())


@pytest.fixture
def deps(console):
    d = AgentDeps(cwd=Path("/tmp"), console=console)
    d.model_override = "qwen3.5:4b"
    return d


class TestEscalationSignals:
    @patch("forge.config.settings", new_callable=_make_settings)
    def test_signals_accumulate(self, mock_s, deps, console):
        esc = ModelEscalator(deps, console, threshold=5.0)
        assert esc.is_active

        esc.add_signal("tool_failure", 1.0, 1)
        esc.add_signal("tool_failure", 1.0, 2)
        assert esc.total_weight == 2.0
        assert not esc.escalated

    @patch("forge.config.settings", new_callable=_make_settings)
    def test_threshold_triggers_escalation(self, mock_s, deps, console):
        esc = ModelEscalator(deps, console, threshold=5.0)

        for i in range(5):
            esc.add_signal("tool_failure", 1.0, i)

        assert esc.escalated
        assert deps.model_override is None  # heavy = None

    @patch("forge.config.settings", new_callable=_make_settings)
    def test_circuit_breaker_contributes_weight(self, mock_s, deps, console):
        esc = ModelEscalator(deps, console, threshold=5.0)

        esc.add_signal("circuit_breaker", 3.0, 1)
        esc.add_signal("tool_failure", 1.0, 2)
        esc.add_signal("tool_failure", 1.0, 3)
        assert esc.escalated

    @patch("forge.config.settings", new_callable=_make_settings)
    def test_below_threshold_no_escalation(self, mock_s, deps, console):
        esc = ModelEscalator(deps, console, threshold=5.0)

        esc.add_signal("tool_failure", 1.0, 1)
        esc.add_signal("tool_failure", 1.0, 2)
        assert not esc.escalated
        assert deps.model_override == "qwen3.5:4b"


class TestEscalationDormant:
    @patch("forge.config.settings", new_callable=_make_settings)
    def test_dormant_on_heavy_model(self, mock_s, deps, console):
        deps.model_override = None  # heavy model
        esc = ModelEscalator(deps, console, threshold=5.0)
        assert not esc.is_active

        esc.add_signal("tool_failure", 10.0, 1)
        assert not esc.escalated
        assert esc.total_weight == 0.0

    @patch("forge.config.settings", new_callable=_make_settings)
    def test_dormant_on_custom_model(self, mock_s, deps, console):
        deps.model_override = "some-custom-model"
        esc = ModelEscalator(deps, console, threshold=5.0)
        assert not esc.is_active


class TestEscalationReset:
    @patch("forge.config.settings", new_callable=_make_settings)
    def test_model_fast_resets(self, mock_s, deps, console):
        esc = ModelEscalator(deps, console, threshold=5.0)

        for i in range(5):
            esc.add_signal("tool_failure", 1.0, i)
        assert esc.escalated

        # User switches back to fast
        deps.model_override = "qwen3.5:4b"
        esc.reset()
        assert not esc.escalated
        assert esc.total_weight == 0.0
        assert esc.is_active
