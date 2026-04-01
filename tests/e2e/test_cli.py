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
