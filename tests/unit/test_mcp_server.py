"""Tests for the Forge MCP server tools."""

import os
import textwrap
from pathlib import Path

import pytest

from forge.mcp_server import (
    _cwd,
    _resolve,
    edit_file,
    list_files,
    read_file,
    run_command,
    search_code,
    write_file,
)

import forge.mcp_server as mcp_mod


@pytest.fixture(autouse=True)
def _use_tmp(tmp_path, monkeypatch):
    """Point the MCP server's working directory at a temp dir."""
    monkeypatch.setattr(mcp_mod, "_cwd", tmp_path)


class TestResolve:
    def test_relative_path(self, tmp_path):
        result = _resolve("foo.py")
        assert result == (tmp_path / "foo.py").resolve()

    def test_absolute_path(self, tmp_path):
        result = _resolve("/tmp/abs.py")
        assert result == Path("/tmp/abs.py").resolve()


class TestReadFile:
    def test_reads_file_with_line_numbers(self, tmp_path):
        (tmp_path / "hello.py").write_text("line1\nline2\nline3\n")
        result = read_file("hello.py")
        assert "hello.py" in result
        assert "3 lines" in result
        assert "line1" in result
        assert "line3" in result

    def test_offset_and_limit(self, tmp_path):
        (tmp_path / "big.txt").write_text("\n".join(f"L{i}" for i in range(20)))
        result = read_file("big.txt", offset=5, limit=3)
        assert "showing lines 6-8" in result
        assert "L5" in result  # 0-indexed line 5

    def test_missing_file(self):
        result = read_file("nope.py")
        assert "Error" in result

    def test_directory_error(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        result = read_file("subdir")
        assert "directory" in result.lower()


class TestWriteFile:
    def test_writes_and_creates_parents(self, tmp_path):
        result = write_file("a/b/c.txt", "hello\nworld\n")
        assert "2 lines" in result
        assert (tmp_path / "a" / "b" / "c.txt").read_text() == "hello\nworld\n"

    def test_overwrites_existing(self, tmp_path):
        (tmp_path / "f.txt").write_text("old")
        write_file("f.txt", "new")
        assert (tmp_path / "f.txt").read_text() == "new"


class TestEditFile:
    def test_replaces_unique_text(self, tmp_path):
        (tmp_path / "e.py").write_text("foo = 1\nbar = 2\n")
        result = edit_file("e.py", "foo = 1", "foo = 42")
        assert "Edited" in result
        assert "foo = 42" in (tmp_path / "e.py").read_text()

    def test_missing_file(self):
        result = edit_file("nope.py", "a", "b")
        assert "Error" in result

    def test_text_not_found(self, tmp_path):
        (tmp_path / "e.py").write_text("hello")
        result = edit_file("e.py", "nope", "yes")
        assert "not found" in result

    def test_duplicate_text_error(self, tmp_path):
        (tmp_path / "e.py").write_text("x\nx\n")
        result = edit_file("e.py", "x", "y")
        assert "2 times" in result


class TestSearchCode:
    def test_finds_pattern(self, tmp_path):
        (tmp_path / "s.py").write_text("def hello():\n    pass\n")
        result = search_code("def hello")
        # ripgrep may not be installed in CI — handle gracefully
        assert "hello" in result or "not installed" in result

    def test_no_matches(self, tmp_path):
        (tmp_path / "s.py").write_text("nothing here\n")
        result = search_code("zzzznotfound")
        assert "No matches" in result or "not installed" in result


class TestListFiles:
    def test_lists_files(self, tmp_path):
        (tmp_path / "a.py").touch()
        (tmp_path / "b.txt").touch()
        result = list_files()
        assert "a.py" in result
        assert "b.txt" in result

    def test_glob_pattern(self, tmp_path):
        (tmp_path / "a.py").touch()
        (tmp_path / "b.txt").touch()
        result = list_files(pattern="*.py")
        assert "a.py" in result
        assert "b.txt" not in result

    def test_skips_hidden_dirs(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").touch()
        (tmp_path / "real.py").touch()
        result = list_files()
        assert "config" not in result
        assert "real.py" in result

    def test_not_a_directory(self, tmp_path):
        (tmp_path / "f.txt").touch()
        result = list_files(path="f.txt")
        assert "Error" in result


class TestRunCommand:
    def test_runs_echo(self):
        result = run_command("echo hello")
        assert "hello" in result
        assert "Exit code: 0" in result

    def test_captures_stderr(self):
        result = run_command("echo err >&2")
        assert "err" in result

    def test_timeout(self):
        result = run_command("sleep 10", timeout=0.5)
        assert "timed out" in result.lower()

    def test_nonzero_exit(self):
        result = run_command("exit 1")
        assert "Exit code: 1" in result
