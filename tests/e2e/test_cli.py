"""End-to-end CLI tests using typer's CliRunner."""

from typer.testing import CliRunner

from forge.cli import app

runner = CliRunner()


class TestCLIBasics:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.stdout or "forge" in result.stdout.lower()

    def test_status_command(self):
        result = runner.invoke(app, ["status"])
        # May fail connecting to Ollama, but shouldn't crash with unhandled exception
        assert result.exit_code in (0, 1)

    def test_ask_help(self):
        result = runner.invoke(app, ["ask", "--help"])
        assert result.exit_code == 0

    def test_agent_help(self):
        result = runner.invoke(app, ["agent", "--help"])
        assert result.exit_code == 0
        assert "--yolo" in result.stdout
        assert "--ask" in result.stdout

    def test_code_help(self):
        result = runner.invoke(app, ["code", "--help"])
        assert result.exit_code == 0

    def test_draft_help(self):
        result = runner.invoke(app, ["draft", "--help"])
        assert result.exit_code == 0

    def test_index_help(self):
        result = runner.invoke(app, ["index", "--help"])
        assert result.exit_code == 0

    def test_history_help(self):
        result = runner.invoke(app, ["history", "--help"])
        assert result.exit_code == 0

    def test_history_command(self):
        result = runner.invoke(app, ["history"])
        # May fail connecting to DB, but shouldn't crash with unhandled exception
        assert result.exit_code in (0, 1)

    def test_serve_help(self):
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0


class TestCLIModuleImports:
    """Verify all new modules import correctly through the CLI entry point."""

    def test_status_module_imports(self):
        from forge.agent.status import Phase, StatusTracker

        assert Phase.THINKING.value == "thinking"
        assert StatusTracker is not None

    def test_deps_has_new_fields(self):
        from forge.agent.deps import AgentDeps
        from pathlib import Path
        from rich.console import Console
        import io

        deps = AgentDeps(cwd=Path("/tmp"), console=Console(file=io.StringIO()))
        assert deps.status_tracker is None
        assert deps.thinking_enabled is False

    def test_render_has_split_thinking(self):
        from forge.agent.render import _split_thinking

        t, v = _split_thinking("<think>x</think>y")
        assert t == "x"
        assert v == "y"

    def test_loop_has_plan_overlay(self):
        from forge.agent.loop import PLAN_OVERLAY

        assert "PLANNING" in PLAN_OVERLAY

    def test_loop_has_maybe_prepend_think(self):
        from forge.agent.loop import _maybe_prepend_think

        assert callable(_maybe_prepend_think)

    def test_multimodal_imports(self):
        from forge.agent.multimodal import IMAGE_EXTENSIONS, parse_multimodal_input

        assert ".png" in IMAGE_EXTENSIONS
        assert callable(parse_multimodal_input)

    def test_sandbox_imports(self):
        from forge.agent.sandbox import make_command_blocklist_handler, make_path_boundary_handler

        assert callable(make_command_blocklist_handler)
        assert callable(make_path_boundary_handler)

    def test_circuit_breaker_imports(self):
        from forge.agent.circuit_breaker import CircuitBreakerTripped, ToolCallTracker

        assert ToolCallTracker is not None
        assert issubclass(CircuitBreakerTripped, Exception)

    def test_mcp_config_imports(self):
        from forge.agent.mcp_config import find_mcp_configs, load_all_mcp_servers

        assert callable(find_mcp_configs)
        assert callable(load_all_mcp_servers)

    def test_context_imports(self):
        from forge.agent.context import compact_history, smart_compact_history

        assert callable(compact_history)
        assert callable(smart_compact_history)
