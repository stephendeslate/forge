"""Tests for agent tool functions."""

import shutil

import pytest
from pathlib import Path
from unittest.mock import MagicMock

has_rg = shutil.which("rg") is not None

from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.permissions import PermissionPolicy
from forge.agent.tools import (
    read_file,
    write_file,
    edit_file,
    run_command,
    search_code,
    list_files,
    _resolve_path,
)


@pytest.fixture
def ctx(tmp_path):
    """Mock RunContext with AgentDeps pointing to tmp_path."""
    mock = MagicMock()
    mock.deps = AgentDeps(
        cwd=tmp_path,
        console=Console(file=None, force_terminal=False, no_color=True),
        permission=PermissionPolicy.YOLO,
    )
    return mock


class TestResolvePath:
    def test_relative_path(self, ctx, tmp_path):
        result = _resolve_path(ctx, "foo.txt")
        assert result == (tmp_path / "foo.txt").resolve()

    def test_absolute_path(self, ctx):
        result = _resolve_path(ctx, "/etc/hosts")
        assert result == Path("/etc/hosts")


class TestReadFile:
    async def test_read_existing_file(self, ctx, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line 1\nline 2\nline 3\n")
        result = await read_file(ctx, "test.txt")
        assert "line 1" in result
        assert "line 2" in result
        assert "3 lines" in result

    async def test_read_nonexistent_file(self, ctx):
        result = await read_file(ctx, "missing.txt")
        assert "Error" in result
        assert "not found" in result.lower()

    async def test_read_with_offset(self, ctx, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("\n".join(f"line {i}" for i in range(1, 11)))
        result = await read_file(ctx, "lines.txt", offset=5, limit=3)
        assert "line 6" in result
        assert "line 8" in result

    async def test_read_with_limit(self, ctx, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line {i}" for i in range(100)))
        result = await read_file(ctx, "big.txt", limit=5)
        assert "line 0" in result
        assert "line 4" in result
        assert "showing lines" in result.lower()

    async def test_read_directory_returns_error(self, ctx, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        result = await read_file(ctx, "subdir")
        assert "Error" in result


class TestWriteFile:
    async def test_write_new_file(self, ctx, tmp_path):
        result = await write_file(ctx, "new.txt", "hello world\n")
        assert "Wrote" in result
        assert (tmp_path / "new.txt").read_text() == "hello world\n"

    async def test_write_creates_parent_dirs(self, ctx, tmp_path):
        result = await write_file(ctx, "a/b/c/deep.txt", "deep content")
        assert "Wrote" in result
        assert (tmp_path / "a/b/c/deep.txt").read_text() == "deep content"

    async def test_write_overwrites_existing(self, ctx, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old")
        await write_file(ctx, "existing.txt", "new")
        assert f.read_text() == "new"


class TestEditFile:
    async def test_edit_replaces_unique_text(self, ctx, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def hello():\n    return 'hello'\n")
        result = await edit_file(ctx, "code.py", "return 'hello'", "return 'goodbye'")
        assert "Edited" in result
        assert "goodbye" in f.read_text()

    async def test_edit_nonexistent_file(self, ctx):
        result = await edit_file(ctx, "missing.py", "a", "b")
        assert "Error" in result
        assert "not found" in result.lower()

    async def test_edit_text_not_found(self, ctx, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("hello world")
        result = await edit_file(ctx, "test.py", "nonexistent text", "replacement")
        assert "Error" in result
        assert "not found" in result.lower()

    async def test_edit_non_unique_text(self, ctx, tmp_path):
        f = tmp_path / "dup.py"
        f.write_text("x = 1\nx = 1\n")
        result = await edit_file(ctx, "dup.py", "x = 1", "x = 2")
        assert "Error" in result
        assert "2 times" in result


class TestRunCommand:
    async def test_run_simple_command(self, ctx):
        result = await run_command(ctx, "echo hello")
        assert "hello" in result
        assert "Exit code: 0" in result

    async def test_run_failing_command(self, ctx):
        result = await run_command(ctx, "false")
        assert "Exit code:" in result

    async def test_run_with_stderr(self, ctx):
        result = await run_command(ctx, "echo err >&2")
        assert "err" in result

    async def test_run_timeout(self, ctx):
        result = await run_command(ctx, "sleep 60", timeout=0.5)
        assert "timed out" in result.lower()

    async def test_run_in_cwd(self, ctx, tmp_path):
        (tmp_path / "marker.txt").write_text("found")
        result = await run_command(ctx, "cat marker.txt")
        assert "found" in result

    async def test_run_truncates_long_output(self, ctx):
        result = await run_command(ctx, "python3 -c \"print('x' * 15000)\"")
        assert "truncated" in result.lower() or len(result) <= 12000


@pytest.mark.skipif(not has_rg, reason="ripgrep (rg) not installed")
class TestSearchCode:
    async def test_search_finds_pattern(self, ctx, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("import os\nimport sys\ndef main():\n    pass\n")
        result = await search_code(ctx, "import", path=".")
        assert "import" in result

    async def test_search_no_matches(self, ctx, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("hello world\n")
        result = await search_code(ctx, "nonexistent_pattern_xyz")
        assert "no matches" in result.lower()

    async def test_search_with_glob_filter(self, ctx, tmp_path):
        (tmp_path / "code.py").write_text("target = 1\n")
        (tmp_path / "data.txt").write_text("target = 2\n")
        result = await search_code(ctx, "target", glob_filter="*.py")
        assert "code.py" in result

    async def test_search_nonexistent_path(self, ctx):
        result = await search_code(ctx, "test", path="nonexistent_dir")
        assert "error" in result.lower() or "not found" in result.lower()


class TestListFiles:
    async def test_list_all_files(self, ctx, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.txt").write_text("")
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "c.py").write_text("")
        result = await list_files(ctx)
        assert "a.py" in result
        assert "b.txt" in result
        assert "c.py" in result

    async def test_list_with_glob_pattern(self, ctx, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.txt").write_text("")
        result = await list_files(ctx, pattern="*.py")
        assert "a.py" in result
        assert "b.txt" not in result

    async def test_list_empty_directory(self, ctx, tmp_path):
        result = await list_files(ctx, pattern="*.rs")
        assert "no files" in result.lower()

    async def test_list_skips_hidden_dirs(self, ctx, tmp_path):
        git = tmp_path / ".git"
        git.mkdir()
        (git / "config").write_text("")
        (tmp_path / "main.py").write_text("")
        result = await list_files(ctx)
        assert "main.py" in result
        assert ".git" not in result

    async def test_list_skips_pycache(self, ctx, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "module.cpython-312.pyc").write_text("")
        (tmp_path / "module.py").write_text("")
        result = await list_files(ctx)
        assert "module.py" in result
        assert "__pycache__" not in result
