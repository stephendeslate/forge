"""Integration tests for cloud recovery and Gemini escalation paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from rich.console import Console

from forge.agent.circuit_breaker import CircuitBreakerTripped
from forge.agent.deps import AgentDeps
from forge.agent.permissions import PermissionPolicy


class TestHandleAgentErrorEscalation:
    """Test _handle_agent_error sets recovery flag when Gemini is available."""

    def test_circuit_breaker_with_gemini_sets_flag(self):
        from forge.agent.loop import _handle_agent_error

        console = Console(file=None, force_terminal=False, no_color=True)
        deps = AgentDeps(
            cwd=Path("/tmp"),
            console=console,
            cloud_reasoning_enabled=True,
        )
        exc = CircuitBreakerTripped("identical calls detected")

        with patch("forge.agent.gemini._ensure_api_key", return_value="test-key"):
            _handle_agent_error(console, exc, deps=deps)

        assert deps._gemini_recovery_pending is True

    def test_circuit_breaker_without_gemini_no_flag(self):
        from forge.agent.loop import _handle_agent_error

        console = Console(file=None, force_terminal=False, no_color=True)
        deps = AgentDeps(
            cwd=Path("/tmp"),
            console=console,
            cloud_reasoning_enabled=False,
        )
        exc = CircuitBreakerTripped("identical calls detected")

        _handle_agent_error(console, exc, deps=deps)

        assert deps._gemini_recovery_pending is False

    def test_circuit_breaker_no_deps_no_crash(self):
        """When deps is None, should print message without crashing."""
        from forge.agent.loop import _handle_agent_error

        console = Console(file=None, force_terminal=False, no_color=True)
        exc = CircuitBreakerTripped("test")

        # Should not raise
        _handle_agent_error(console, exc, deps=None)

    def test_non_circuit_breaker_error_no_flag(self):
        from forge.agent.loop import _handle_agent_error

        console = Console(file=None, force_terminal=False, no_color=True)
        deps = AgentDeps(
            cwd=Path("/tmp"),
            console=console,
            cloud_reasoning_enabled=True,
        )

        _handle_agent_error(console, RuntimeError("some error"), deps=deps)
        assert deps._gemini_recovery_pending is False


class TestCloudRecoveryFunction:
    """Test _cloud_recovery() with mocked Agent + _run_with_status."""

    @pytest.mark.asyncio
    async def test_recovery_calls_fresh_agent(self):
        """Recovery should create a fresh agent without conversation history."""
        from forge.agent.loop import _cloud_recovery

        console = Console(file=None, force_terminal=False, no_color=True)
        deps = AgentDeps(
            cwd=Path("/tmp"),
            console=console,
            cloud_reasoning_enabled=True,
        )

        with patch("forge.agent.gemini._ensure_api_key", return_value="test-key"), \
             patch("forge.agent.loop.Agent") as MockAgent, \
             patch("forge.agent.loop._run_with_status", new_callable=AsyncMock) as mock_run:
            MockAgent.return_value = MagicMock()
            mock_run.return_value = [MagicMock()]

            result = await _cloud_recovery("fix the bug", deps, console, turn_number=1)

        assert result is not None
        # Verify _run_with_status was called with message_history=None (fresh agent)
        call_args = mock_run.call_args
        assert call_args[0][3] is None  # 4th positional arg is message_history

    @pytest.mark.asyncio
    async def test_recovery_resets_circuit_breaker(self):
        """Recovery should reset the circuit breaker before retrying."""
        from forge.agent.loop import _cloud_recovery
        from forge.agent.circuit_breaker import ToolCallTracker

        console = Console(file=None, force_terminal=False, no_color=True)
        cb = ToolCallTracker()
        deps = AgentDeps(
            cwd=Path("/tmp"),
            console=console,
            cloud_reasoning_enabled=True,
            circuit_breaker=cb,
        )

        with patch("forge.agent.gemini._ensure_api_key", return_value="test-key"), \
             patch("forge.agent.loop.Agent") as MockAgent, \
             patch("forge.agent.loop._run_with_status", new_callable=AsyncMock) as mock_run:
            MockAgent.return_value = MagicMock()
            mock_run.return_value = [MagicMock()]

            await _cloud_recovery("do something", deps, console, turn_number=1)

        # Circuit breaker should have been reset (state cleared)
        assert len(cb._history) == 0

    @pytest.mark.asyncio
    async def test_recovery_fallback_on_primary_failure(self):
        """If primary model fails, should try fallback model."""
        from forge.agent.loop import _cloud_recovery

        console = Console(file=None, force_terminal=False, no_color=True)
        deps = AgentDeps(
            cwd=Path("/tmp"),
            console=console,
            cloud_reasoning_enabled=True,
        )

        call_count = 0

        async def mock_run_with_status(agent, prompt, deps, history, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("primary failed")
            return [MagicMock()]

        with patch("forge.agent.gemini._ensure_api_key", return_value="test-key"), \
             patch("forge.agent.loop.Agent") as MockAgent, \
             patch("forge.agent.loop._run_with_status", side_effect=mock_run_with_status):
            MockAgent.return_value = MagicMock()
            result = await _cloud_recovery("fix it", deps, console, turn_number=1)

        assert result is not None
        assert call_count == 2  # tried primary, then fallback

    @pytest.mark.asyncio
    async def test_recovery_returns_none_on_total_failure(self):
        """If all models fail, should return None."""
        from forge.agent.loop import _cloud_recovery

        console = Console(file=None, force_terminal=False, no_color=True)
        deps = AgentDeps(
            cwd=Path("/tmp"),
            console=console,
            cloud_reasoning_enabled=True,
        )

        async def always_fail(agent, prompt, deps, history, **kwargs):
            raise ConnectionError("all failed")

        with patch("forge.agent.gemini._ensure_api_key", return_value="test-key"), \
             patch("forge.agent.loop.Agent") as MockAgent, \
             patch("forge.agent.loop._run_with_status", side_effect=always_fail):
            MockAgent.return_value = MagicMock()
            result = await _cloud_recovery("fix it", deps, console, turn_number=1)

        assert result is None


class TestPlanGeminiIntegration:
    """Test that _plan_and_execute() picks Gemini model when available."""

    @pytest.mark.asyncio
    async def test_plan_uses_gemini_when_available(self):
        """When cloud reasoning is on and key exists, plan agent should use Gemini model."""
        from forge.agent.loop import _plan_and_execute

        console = Console(file=None, force_terminal=False, no_color=True)
        deps = AgentDeps(
            cwd=Path("/tmp"),
            console=console,
            cloud_reasoning_enabled=True,
        )

        with patch("forge.agent.gemini._ensure_api_key", return_value="test-key"), \
             patch("forge.agent.loop.Agent") as MockAgent:

            mock_plan_agent = MagicMock()
            mock_plan_result = MagicMock()
            mock_plan_result.output = "Plan: do the thing"

            async def mock_run(**kwargs):
                return mock_plan_result

            mock_plan_agent.run = mock_run
            MockAgent.return_value = mock_plan_agent

            with patch("forge.agent.loop.asyncio.get_running_loop") as mock_loop:
                mock_executor = AsyncMock(return_value="n")
                mock_loop.return_value.run_in_executor = mock_executor

                await _plan_and_execute(MagicMock(), "refactor config", deps, None)

            # Check Agent was called with a Gemini model string
            agent_call = MockAgent.call_args
            model_used = agent_call.kwargs.get("model", agent_call.args[0] if agent_call.args else None)
            assert model_used is not None
            assert "google-gla:" in str(model_used)

    @pytest.mark.asyncio
    async def test_plan_falls_back_to_local_without_key(self):
        """When no Gemini key, plan should fall back to local model."""
        from forge.agent.loop import _plan_and_execute

        console = Console(file=None, force_terminal=False, no_color=True)
        deps = AgentDeps(
            cwd=Path("/tmp"),
            console=console,
            cloud_reasoning_enabled=True,  # enabled but no key
        )

        with patch("forge.agent.gemini._ensure_api_key", return_value=None), \
             patch("forge.agent.loop.Agent") as MockAgent:

            mock_plan_agent = MagicMock()
            mock_plan_result = MagicMock()
            mock_plan_result.output = "Local plan"

            async def mock_run(**kwargs):
                return mock_plan_result

            mock_plan_agent.run = mock_run
            MockAgent.return_value = mock_plan_agent

            with patch("forge.agent.loop.asyncio.get_running_loop") as mock_loop:
                mock_executor = AsyncMock(return_value="n")
                mock_loop.return_value.run_in_executor = mock_executor

                await _plan_and_execute(MagicMock(), "plan something", deps, None)

            agent_call = MockAgent.call_args
            model_used = agent_call.kwargs.get("model", agent_call.args[0] if agent_call.args else None)
            assert "ollama:" in str(model_used)
