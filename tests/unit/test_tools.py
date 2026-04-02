"""Tests for agent tool functions."""

import shutil

import pytest
from pathlib import Path
from unittest.mock import MagicMock

has_rg = shutil.which("rg") is not None

from pydantic_ai import ModelRetry, Tool
from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.permissions import PermissionPolicy
from forge.agent.tools import (
    ALL_TOOLS,
    read_file,
    write_file,
    edit_file,
    run_command,
    search_code,
    list_files,
    rag_search,
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

    async def test_read_nonexistent_file_raises_model_retry(self, ctx):
        with pytest.raises(ModelRetry, match="File not found"):
            await read_file(ctx, "missing.txt")

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

    async def test_read_directory_raises_model_retry(self, ctx, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        with pytest.raises(ModelRetry, match="directory"):
            await read_file(ctx, "subdir")


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

    async def test_edit_nonexistent_file_raises_model_retry(self, ctx):
        with pytest.raises(ModelRetry, match="File not found"):
            await edit_file(ctx, "missing.py", "a", "b")

    async def test_edit_text_not_found_raises_model_retry(self, ctx, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("hello world")
        with pytest.raises(ModelRetry, match="old_text not found"):
            await edit_file(ctx, "test.py", "nonexistent text", "replacement")

    async def test_edit_non_unique_text_raises_model_retry(self, ctx, tmp_path):
        f = tmp_path / "dup.py"
        f.write_text("x = 1\nx = 1\n")
        with pytest.raises(ModelRetry, match="appears 2 times"):
            await edit_file(ctx, "dup.py", "x = 1", "x = 2")


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
        result = await run_command(ctx, "python3 -c \"print('x' * 60000)\"")
        assert "truncated" in result.lower() or len(result) <= 55000


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


class TestModelRetry:
    """Verify retryable errors raise ModelRetry."""

    async def test_edit_nonexistent_raises_retry(self, ctx):
        """Editing a nonexistent file raises ModelRetry (retryable)."""
        with pytest.raises(ModelRetry, match="File not found"):
            await edit_file(ctx, "nonexistent.py", "a", "b")

    async def test_permission_blocked_via_hooks(self, ctx, tmp_path):
        """Permission denial via hooks raises ModelRetry."""
        from forge.agent.hooks import HookAction, HookRegistry, HookResult, PreToolUse, with_hooks

        registry = HookRegistry()

        async def blocker(event):
            if event.tool_name == "write_file":
                return HookResult(action=HookAction.BLOCK, message="Permission denied by user.")
            return HookResult()

        registry.on(PreToolUse, blocker)
        ctx.deps.hook_registry = registry

        # Call the hooked write_file directly
        hooked_write = with_hooks(write_file)
        with pytest.raises(ModelRetry, match="Permission denied"):
            await hooked_write(ctx, file_path="test.py", content="x")


class TestParallelToolMarking:
    """Verify ALL_TOOLS has correct sequential/parallel marking."""

    def test_read_only_tools_are_not_sequential(self):
        non_seq = [t for t in ALL_TOOLS if isinstance(t, Tool) and not t.sequential]
        names = {t.name for t in non_seq}
        assert "read_file" in names
        assert "search_code" in names
        assert "list_files" in names
        assert "web_search" in names
        assert "web_fetch" in names

    def test_write_tools_are_sequential(self):
        seq_tools = [t for t in ALL_TOOLS if isinstance(t, Tool) and t.sequential]
        names = {t.name for t in seq_tools}
        assert "write_file" in names
        assert "edit_file" in names
        assert "run_command" in names

    def test_no_write_tool_is_parallel(self):
        for t in ALL_TOOLS:
            if isinstance(t, Tool) and t.name in ("write_file", "edit_file", "run_command"):
                assert t.sequential is True


class TestRagSearch:
    """Test rag_search tool graceful fallback."""

    async def test_rag_unavailable_returns_fallback(self, ctx):
        """When RAG is not configured, returns a helpful message."""
        ctx.deps.rag_db = None
        ctx.deps.rag_project = None
        result = await rag_search(ctx, "how does routing work")
        assert "unavailable" in result.lower()
        assert "search_code" in result
