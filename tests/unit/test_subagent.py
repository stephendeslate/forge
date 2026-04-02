"""Tests for sub-agent spawning and delegation."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from forge.agent.hooks import HookRegistry, PostToolUse
from forge.agent.subagent import (
    FAILURE_PATTERNS,
    MergeResult,
    SubagentResult,
    SUBAGENT_TOOLS,
    SUBAGENT_SYSTEM,
    _validate_output,
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


class TestValidateOutput:
    def test_success_output(self):
        r = SubagentResult(output="Changed 3 files", worktree=None, messages=[], success=True)
        result = _validate_output(r)
        assert result.success is True

    def test_empty_output_fails(self):
        r = SubagentResult(output="", worktree=None, messages=[], success=True)
        result = _validate_output(r)
        assert result.success is False

    def test_whitespace_only_fails(self):
        r = SubagentResult(output="   \n  ", worktree=None, messages=[], success=True)
        result = _validate_output(r)
        assert result.success is False

    def test_none_output_fails(self):
        r = SubagentResult(output="", worktree=None, messages=[], success=True)
        result = _validate_output(r)
        assert result.success is False

    def test_task_failure_detected(self):
        """Patterns that indicate genuine task failure should be caught."""
        for indicator in [
            "could not complete the task",
            "Failed to accomplish the refactoring",
            "Traceback (most recent call last)",
            "error: something went wrong",
            "fatal: not a git repository",
            "unable to finish the implementation",
            "timed out after 30 seconds waiting for response",
        ]:
            r = SubagentResult(output=indicator, worktree=None, messages=[], success=True)
            result = _validate_output(r)
            assert result.success is False, f"Should detect: {indicator}"

    def test_legitimate_output_not_flagged(self):
        """Output containing error-related words in non-failure context should pass."""
        for output in [
            "I cannot find any issues with the code",
            "Error handling was improved in the module",
            "Previously failed to connect, but now it works",
            "The function handles the 'could not open file' case gracefully",
            "Added error recovery logic to the parser",
            "Fixed the traceback formatting in the logger",
            "Unable to reproduce the reported bug — code looks correct",
        ]:
            r = SubagentResult(output=output, worktree=None, messages=[], success=True)
            result = _validate_output(r)
            assert result.success is True, f"Should NOT flag: {output}"

    def test_already_failed_stays_failed(self):
        r = SubagentResult(output="fine output", worktree=None, messages=[], success=False)
        result = _validate_output(r)
        assert result.success is False  # doesn't override existing failure


class TestMergeResult:
    def test_fields(self):
        r = MergeResult(merged=True, conflict=False, message="ok")
        assert r.merged is True
        assert r.conflict is False


class TestHookInheritance:
    def test_get_handlers_empty(self):
        reg = HookRegistry()
        assert reg.get_handlers(PostToolUse) == []

    def test_get_handlers_returns_copy(self):
        reg = HookRegistry()
        handler = MagicMock()
        reg.on(PostToolUse, handler, priority=5)

        handlers = reg.get_handlers(PostToolUse)
        assert len(handlers) == 1
        assert handlers[0] == (5, handler)

        # Modifying the copy shouldn't affect the registry
        handlers.clear()
        assert len(reg.get_handlers(PostToolUse)) == 1

    def test_handlers_propagate_to_child(self):
        parent = HookRegistry()
        handler = MagicMock()
        parent.on(PostToolUse, handler, priority=10)

        child = HookRegistry()
        for priority, h in parent.get_handlers(PostToolUse):
            child.on(PostToolUse, h, priority=priority)

        assert len(child.get_handlers(PostToolUse)) == 1
        assert child.get_handlers(PostToolUse)[0] == (10, handler)
