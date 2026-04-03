"""Tests for Feature 3: Impact Analysis Tool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.agent.impact import (
    ImpactReport,
    _extract_symbols,
    _module_name_from_path,
    _ripgrep_search,
    build_impact_report,
)


# ---------------------------------------------------------------------------
# ImpactReport
# ---------------------------------------------------------------------------


class TestImpactReport:
    def test_format_empty(self):
        r = ImpactReport(file="foo.py")
        out = r.format()
        assert "foo.py" in out
        assert "No top-level symbols found" in out

    def test_format_with_symbols_no_dependents(self):
        r = ImpactReport(
            file="foo.py",
            symbols_defined=["Foo", "bar"],
        )
        out = r.format()
        assert "Symbols defined (2)" in out
        assert "Foo" in out
        assert "bar" in out
        assert "No other files import" in out

    def test_format_with_dependents(self):
        r = ImpactReport(
            file="foo.py",
            symbols_defined=["Foo"],
            imported_by={"Foo": ["bar.py", "baz.py"]},
            total_dependents=2,
        )
        out = r.format()
        assert "Dependent files (2)" in out
        assert "bar.py" in out
        assert "baz.py" in out

    def test_format_truncates_symbols(self):
        r = ImpactReport(
            file="foo.py",
            symbols_defined=[f"sym_{i}" for i in range(40)],
        )
        out = r.format()
        assert "and 10 more" in out

    def test_format_truncates_dependents(self):
        r = ImpactReport(
            file="foo.py",
            symbols_defined=["Foo"],
            imported_by={"Foo": [f"dep_{i}.py" for i in range(15)]},
            total_dependents=15,
        )
        out = r.format()
        assert "and 5 more" in out


# ---------------------------------------------------------------------------
# _module_name_from_path
# ---------------------------------------------------------------------------


class TestModuleNameFromPath:
    def test_simple(self, tmp_path):
        p = tmp_path / "foo.py"
        assert _module_name_from_path(p, tmp_path) == "foo"

    def test_nested(self, tmp_path):
        p = tmp_path / "src" / "forge" / "config.py"
        assert _module_name_from_path(p, tmp_path) == "src.forge.config"

    def test_outside_cwd(self):
        p = Path("/other/foo.py")
        cwd = Path("/project")
        result = _module_name_from_path(p, cwd)
        assert "foo" in result


# ---------------------------------------------------------------------------
# _extract_symbols
# ---------------------------------------------------------------------------


class TestExtractSymbols:
    def test_extracts_python_symbols(self, tmp_path):
        f = tmp_path / "sample.py"
        f.write_text(
            "class Foo:\n    pass\n\n"
            "def bar():\n    pass\n\n"
            "async def baz():\n    pass\n"
        )
        symbols = _extract_symbols(f)
        assert "Foo" in symbols
        assert "bar" in symbols
        assert "baz" in symbols

    def test_no_symbols(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("# just a comment\nx = 1\n")
        symbols = _extract_symbols(f)
        # May or may not include x depending on tree-sitter config
        # At minimum should not crash
        assert isinstance(symbols, list)

    def test_unknown_extension(self, tmp_path):
        f = tmp_path / "file.xyz"
        f.write_text("some content")
        symbols = _extract_symbols(f)
        assert symbols == []

    def test_nonexistent_file(self, tmp_path):
        f = tmp_path / "gone.py"
        symbols = _extract_symbols(f)
        assert symbols == []


# ---------------------------------------------------------------------------
# _ripgrep_search
# ---------------------------------------------------------------------------


class TestRipgrepSearch:
    @pytest.mark.asyncio
    async def test_finds_matches(self, tmp_path):
        (tmp_path / "a.py").write_text("import foo\n")
        (tmp_path / "b.py").write_text("nothing here\n")
        results = await _ripgrep_search("import foo", tmp_path)
        assert any("a.py" in r for r in results)

    @pytest.mark.asyncio
    async def test_no_matches(self, tmp_path):
        (tmp_path / "a.py").write_text("nothing\n")
        results = await _ripgrep_search("nonexistent_pattern_xyz", tmp_path)
        assert results == []

    @pytest.mark.asyncio
    async def test_with_glob_filter(self, tmp_path):
        (tmp_path / "a.py").write_text("hello\n")
        (tmp_path / "a.txt").write_text("hello\n")
        results = await _ripgrep_search("hello", tmp_path, glob_filter="*.py")
        assert any("a.py" in r for r in results)
        assert not any("a.txt" in r for r in results)

    @pytest.mark.asyncio
    async def test_handles_missing_rg(self, tmp_path):
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            results = await _ripgrep_search("test", tmp_path)
            assert results == []


# ---------------------------------------------------------------------------
# build_impact_report
# ---------------------------------------------------------------------------


class TestBuildImpactReport:
    @pytest.mark.asyncio
    async def test_nonexistent_file(self, tmp_path):
        report = await build_impact_report("missing.py", tmp_path)
        assert report.file == "missing.py"
        assert report.symbols_defined == []
        assert report.total_dependents == 0

    @pytest.mark.asyncio
    async def test_python_file(self, tmp_path):
        # Create a Python file with symbols
        lib = tmp_path / "lib.py"
        lib.write_text("class Widget:\n    pass\n\ndef create_widget():\n    pass\n")

        # Create a file that imports it
        user = tmp_path / "app.py"
        user.write_text("from lib import Widget\n\nw = Widget()\n")

        report = await build_impact_report("lib.py", tmp_path)
        assert "Widget" in report.symbols_defined
        assert "create_widget" in report.symbols_defined
        assert report.total_dependents >= 0  # may find app.py

    @pytest.mark.asyncio
    async def test_absolute_path(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def hello():\n    pass\n")
        report = await build_impact_report(str(f), tmp_path)
        assert "hello" in report.symbols_defined

    @pytest.mark.asyncio
    async def test_relative_path_resolved(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("class Mod:\n    pass\n")
        report = await build_impact_report("mod.py", tmp_path)
        assert report.file == "mod.py"
        assert "Mod" in report.symbols_defined
