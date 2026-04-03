"""Tests for project detection and instruction loading."""

from pathlib import Path

from forge.core.project import (
    INSTRUCTION_FILES,
    MAX_INSTRUCTION_FILE_CHARS,
    MAX_TOTAL_INSTRUCTION_CHARS,
    build_project_context,
    detect_project_type,
    load_project_instructions,
)


class TestDetectProjectType:
    def test_python_project(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        assert "Python" in detect_project_type(tmp_path)

    def test_node_project(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "test"}')
        assert "Node.js" in detect_project_type(tmp_path)

    def test_multiple_types(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]")
        (tmp_path / "package.json").write_text("{}")
        result = detect_project_type(tmp_path)
        assert "Python" in result
        assert "Node.js" in result

    def test_no_manifest(self, tmp_path):
        assert detect_project_type(tmp_path) == ""


class TestLoadProjectInstructions:
    def test_loads_claude_md(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# Instructions\nDo stuff")
        result = load_project_instructions(tmp_path)
        assert "Do stuff" in result

    def test_loads_forge_md(self, tmp_path):
        (tmp_path / "FORGE.md").write_text("# Forge\nForge instructions")
        result = load_project_instructions(tmp_path)
        assert "Forge instructions" in result

    def test_loads_dotforge_instructions(self, tmp_path):
        forge_dir = tmp_path / ".forge"
        forge_dir.mkdir()
        (forge_dir / "instructions.md").write_text("Dot forge instructions")
        result = load_project_instructions(tmp_path)
        assert "Dot forge instructions" in result

    def test_ancestor_chain(self, tmp_path):
        """Instructions from parent directories are also loaded."""
        (tmp_path / "CLAUDE.md").write_text("Parent instructions")
        child = tmp_path / "sub" / "project"
        child.mkdir(parents=True)
        (child / "CLAUDE.md").write_text("Child instructions")
        result = load_project_instructions(child)
        assert "Child instructions" in result
        assert "Parent instructions" in result

    def test_deduplicates_by_content(self, tmp_path):
        """Same content at different levels is not duplicated."""
        (tmp_path / "CLAUDE.md").write_text("Same content")
        child = tmp_path / "sub"
        child.mkdir()
        (child / "CLAUDE.md").write_text("Same content")
        result = load_project_instructions(child)
        assert result.count("Same content") == 1

    def test_truncates_large_files(self, tmp_path):
        large = "x" * (MAX_INSTRUCTION_FILE_CHARS + 1000)
        (tmp_path / "CLAUDE.md").write_text(large)
        result = load_project_instructions(tmp_path)
        assert "truncated" in result

    def test_total_budget_respected(self, tmp_path):
        """Total instruction content is capped."""
        content = "y" * (MAX_INSTRUCTION_FILE_CHARS - 10)
        # Create enough files to exceed total budget
        for name in INSTRUCTION_FILES:
            p = tmp_path / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content + name)  # Different content for dedup

        result = load_project_instructions(tmp_path)
        assert len(result) <= MAX_TOTAL_INSTRUCTION_CHARS + 500  # Allow header overhead

    def test_empty_file_skipped(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("")
        assert load_project_instructions(tmp_path) == ""

    def test_no_instruction_files(self, tmp_path):
        assert load_project_instructions(tmp_path) == ""


class TestBuildProjectContext:
    def test_includes_cwd(self, tmp_path):
        result = build_project_context(tmp_path)
        assert str(tmp_path) in result

    def test_includes_project_type(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]")
        result = build_project_context(tmp_path)
        assert "Python" in result

    def test_includes_instructions(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("Build stuff")
        result = build_project_context(tmp_path)
        assert "Build stuff" in result
