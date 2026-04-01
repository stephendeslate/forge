"""Integration tests for the agent loop module."""

import os
from unittest.mock import patch

from forge.agent.loop import (
    _detect_project_type,
    _load_project_instructions,
    _ensure_ollama_env,
)


class TestDetectProjectType:
    def test_python_project(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        assert "Python" in _detect_project_type(tmp_path)

    def test_node_project(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "test"}')
        assert "Node.js" in _detect_project_type(tmp_path)

    def test_rust_project(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n')
        assert "Rust" in _detect_project_type(tmp_path)

    def test_go_project(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/test\n")
        assert "Go" in _detect_project_type(tmp_path)

    def test_multi_language_project(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "package.json").write_text("{}")
        result = _detect_project_type(tmp_path)
        assert "Python" in result
        assert "Node.js" in result

    def test_no_manifest(self, tmp_path):
        assert _detect_project_type(tmp_path) == ""


class TestLoadProjectInstructions:
    def test_loads_claude_md(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# Instructions\nDo the thing.\n")
        result = _load_project_instructions(tmp_path)
        assert "Do the thing" in result
        assert "CLAUDE.md" in result

    def test_loads_forge_md(self, tmp_path):
        (tmp_path / "FORGE.md").write_text("# Forge\nForge instructions.\n")
        result = _load_project_instructions(tmp_path)
        assert "Forge instructions" in result

    def test_priority_order(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("CLAUDE rules")
        (tmp_path / "FORGE.md").write_text("FORGE rules")
        result = _load_project_instructions(tmp_path)
        assert "CLAUDE rules" in result

    def test_no_instructions(self, tmp_path):
        assert _load_project_instructions(tmp_path) == ""

    def test_empty_file_skipped(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("   \n  \n")
        assert _load_project_instructions(tmp_path) == ""

    def test_dotforge_instructions(self, tmp_path):
        d = tmp_path / ".forge"
        d.mkdir()
        (d / "instructions.md").write_text("Custom forge instructions")
        result = _load_project_instructions(tmp_path)
        assert "Custom forge instructions" in result


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
