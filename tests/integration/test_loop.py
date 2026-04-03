"""Integration tests for the agent loop module."""

import os
from unittest.mock import patch

from forge.agent.loop import _ensure_ollama_env, _maybe_prepend_think, _run_with_status, create_agent, AGENT_SYSTEM
from forge.agent.deps import AgentDeps
from forge.config import settings
from forge.core.project import detect_project_type, load_project_instructions

from pathlib import Path
from rich.console import Console


class TestDetectProjectType:
    def test_python_project(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        assert "Python" in detect_project_type(tmp_path)

    def test_node_project(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "test"}')
        assert "Node.js" in detect_project_type(tmp_path)

    def test_rust_project(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n')
        assert "Rust" in detect_project_type(tmp_path)

    def test_go_project(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/test\n")
        assert "Go" in detect_project_type(tmp_path)

    def test_multi_language_project(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "package.json").write_text("{}")
        result = detect_project_type(tmp_path)
        assert "Python" in result
        assert "Node.js" in result

    def test_no_manifest(self, tmp_path):
        assert detect_project_type(tmp_path) == ""


class TestLoadProjectInstructions:
    def test_loads_claude_md(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# Instructions\nDo the thing.\n")
        result = load_project_instructions(tmp_path)
        assert "Do the thing" in result
        assert "CLAUDE.md" in result

    def test_loads_forge_md(self, tmp_path):
        (tmp_path / "FORGE.md").write_text("# Forge\nForge instructions.\n")
        result = load_project_instructions(tmp_path)
        assert "Forge instructions" in result

    def test_priority_order(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("CLAUDE rules")
        (tmp_path / "FORGE.md").write_text("FORGE rules")
        result = load_project_instructions(tmp_path)
        assert "CLAUDE rules" in result

    def test_no_instructions(self, tmp_path):
        assert load_project_instructions(tmp_path) == ""

    def test_empty_file_skipped(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("   \n  \n")
        assert load_project_instructions(tmp_path) == ""

    def test_dotforge_instructions(self, tmp_path):
        d = tmp_path / ".forge"
        d.mkdir()
        (d / "instructions.md").write_text("Custom forge instructions")
        result = load_project_instructions(tmp_path)
        assert "Custom forge instructions" in result


class TestAgentSystemPrompt:
    """Verify the system prompt instructs the model to synthesize tool results."""

    def test_system_prompt_mentions_web_tools(self):
        assert "web search" in AGENT_SYSTEM.lower()
        assert "web fetch" in AGENT_SYSTEM.lower()

    def test_system_prompt_has_synthesis_instruction(self):
        """Model must be told to synthesize answers from tool results, not dump raw output."""
        assert "synthesize" in AGENT_SYSTEM.lower() or "raw data" in AGENT_SYSTEM.lower()
        assert "never dump raw" in AGENT_SYSTEM.lower()

    def test_system_prompt_has_answer_guidance(self):
        """Model must be told to answer the user's question based on tool results."""
        assert "answer" in AGENT_SYSTEM.lower()
        # Should mention using tools to gather info, then responding
        assert "clear answer" in AGENT_SYSTEM.lower() or "natural answer" in AGENT_SYSTEM.lower()

    def test_system_prompt_has_web_research_rules(self):
        """System prompt must contain explicit web research rules section."""
        assert "## Web research rules" in AGENT_SYSTEM
        assert "snippets are often enough" in AGENT_SYSTEM.lower()
        assert "budget" in AGENT_SYSTEM.lower()
        assert "never re-fetch" in AGENT_SYSTEM.lower()
        assert "never fetch more than 3" in AGENT_SYSTEM.lower()

    def test_request_limit_is_configured(self):
        """Request limit should be set from settings to catch runaway loops."""
        import inspect
        source = inspect.getsource(_run_with_status)
        assert "request_limit=" in source
        assert "settings.agent.request_limit" in source


class TestEnsureOllamaEnv:
    def test_sets_env_var(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OLLAMA_BASE_URL", None)
            _ensure_ollama_env()
            assert "OLLAMA_BASE_URL" in os.environ
            assert os.environ["OLLAMA_BASE_URL"].endswith("/v1")

    def test_does_not_override_existing(self):
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://custom:1234/v1"}):
            _ensure_ollama_env()
            assert os.environ["OLLAMA_BASE_URL"] == "http://custom:1234/v1"


class TestCreateAgent:
    def test_default_model(self):
        agent = create_agent()
        assert agent.model.model_name == settings.ollama.heavy_model

    def test_custom_model(self):
        agent = create_agent(model="test-model:latest")
        assert agent.model.model_name == "test-model:latest"

    def test_retries_set(self):
        agent = create_agent()
        assert agent._max_tool_retries == 3


class TestModelSwitching:
    def test_model_override_on_deps(self):
        deps = AgentDeps(cwd=Path("/tmp"), console=Console(file=None))
        assert deps.model_override is None
        deps.model_override = "fast-model:latest"
        assert deps.model_override == "fast-model:latest"


class TestPlanTracking:
    def test_active_plan_on_deps(self):
        deps = AgentDeps(cwd=Path("/tmp"), console=Console(file=None))
        assert deps.active_plan is None
        deps.active_plan = "Step 1: Do X\nStep 2: Do Y"
        assert "Step 1" in deps.active_plan

    def test_plan_cleared(self):
        deps = AgentDeps(cwd=Path("/tmp"), console=Console(file=None))
        deps.active_plan = "some plan"
        deps.active_plan = None
        assert deps.active_plan is None


class TestMaybePrependThink:
    def test_thinking_disabled(self):
        deps = AgentDeps(
            cwd=Path("/tmp"),
            console=Console(file=None),
            thinking_enabled=False,
        )
        assert _maybe_prepend_think("hello", deps) == "hello"

    def test_thinking_enabled(self):
        deps = AgentDeps(
            cwd=Path("/tmp"),
            console=Console(file=None),
            thinking_enabled=True,
        )
        result = _maybe_prepend_think("hello", deps)
        assert result.startswith("/think\n")
        assert result.endswith("hello")
