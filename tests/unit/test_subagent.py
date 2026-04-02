"""Tests for sub-agent spawning and delegation."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from forge.agent.subagent import (
    SubagentResult,
    SUBAGENT_TOOLS,
    SUBAGENT_SYSTEM,
    run_subagent,
)


class TestSubagentTools:
    def test_has_core_tools(self):
        names = {t.name for t in SUBAGENT_TOOLS}
        assert "read_file" in names
        assert "write_file" in names
        assert "edit_file" in names
        assert "run_command" in names
        assert "search_code" in names
        assert "list_files" in names

    def test_no_web_or_memory_tools(self):
        names = {t.name for t in SUBAGENT_TOOLS}
        assert "web_search" not in names
        assert "web_fetch" not in names
        assert "save_memory" not in names
        assert "recall_memories" not in names
        assert "delegate" not in names

    def test_write_tools_are_sequential(self):
        for t in SUBAGENT_TOOLS:
            if t.name in ("write_file", "edit_file", "run_command"):
                assert t.sequential is True


class TestSubagentSystem:
    def test_system_prompt_exists(self):
        assert "sub-agent" in SUBAGENT_SYSTEM.lower()
        assert "task" in SUBAGENT_SYSTEM.lower()


class TestSubagentResult:
    def test_result_fields(self):
        r = SubagentResult(
            output="done",
            worktree=None,
            messages=[],
            success=True,
        )
        assert r.output == "done"
        assert r.success is True
        assert r.worktree is None
        assert r.messages == []
