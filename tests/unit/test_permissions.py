"""Tests for the permission system."""

import pytest
from unittest.mock import patch, AsyncMock

from rich.console import Console

from forge.agent.permissions import (
    PermissionPolicy,
    SAFE_TOOLS,
    DANGEROUS_TOOLS,
    check_permission,
    _summarize_call,
)


class TestPermissionPolicy:
    def test_values(self):
        assert PermissionPolicy.AUTO.value == "auto"
        assert PermissionPolicy.ASK.value == "ask"
        assert PermissionPolicy.YOLO.value == "yolo"


class TestToolCategories:
    def test_safe_tools(self):
        assert "read_file" in SAFE_TOOLS
        assert "search_code" in SAFE_TOOLS
        assert "list_files" in SAFE_TOOLS

    def test_dangerous_tools(self):
        assert "write_file" in DANGEROUS_TOOLS
        assert "edit_file" in DANGEROUS_TOOLS
        assert "run_command" in DANGEROUS_TOOLS

    def test_no_overlap(self):
        assert SAFE_TOOLS.isdisjoint(DANGEROUS_TOOLS)


class TestCheckPermission:
    @pytest.fixture
    def console(self):
        return Console(file=None, force_terminal=False, no_color=True)

    async def test_yolo_always_allows(self, console):
        assert await check_permission(console, PermissionPolicy.YOLO, "run_command", {"command": "rm -rf /"})
        assert await check_permission(console, PermissionPolicy.YOLO, "write_file", {"file_path": "x"})

    async def test_auto_allows_safe_tools(self, console):
        for tool in SAFE_TOOLS:
            assert await check_permission(console, PermissionPolicy.AUTO, tool, {})

    async def test_auto_prompts_for_dangerous_tools(self, console):
        with patch("forge.agent.permissions._prompt_user", new_callable=AsyncMock, return_value=False):
            result = await check_permission(
                console, PermissionPolicy.AUTO, "run_command", {"command": "ls"}
            )
            assert result is False

    async def test_auto_allows_when_user_approves(self, console):
        with patch("forge.agent.permissions._prompt_user", new_callable=AsyncMock, return_value=True):
            result = await check_permission(
                console, PermissionPolicy.AUTO, "write_file", {"file_path": "x", "content": "y"}
            )
            assert result is True

    async def test_ask_prompts_for_everything(self, console):
        with patch("forge.agent.permissions._prompt_user", new_callable=AsyncMock, return_value=False):
            result = await check_permission(
                console, PermissionPolicy.ASK, "read_file", {"file_path": "x"}
            )
            assert result is False


class TestSummarizeCall:
    def test_write_file(self):
        summary = _summarize_call("write_file", {"file_path": "test.py", "content": "line1\nline2\n"})
        assert "test.py" in summary

    def test_edit_file(self):
        summary = _summarize_call("edit_file", {"file_path": "main.py"})
        assert "main.py" in summary

    def test_run_command(self):
        summary = _summarize_call("run_command", {"command": "make test"})
        assert "make test" in summary

    def test_read_file(self):
        summary = _summarize_call("read_file", {"file_path": "config.toml"})
        assert "config.toml" in summary

    def test_search_code(self):
        summary = _summarize_call("search_code", {"pattern": "TODO"})
        assert "TODO" in summary

    def test_unknown_tool(self):
        assert _summarize_call("unknown_tool", {}) == ""
