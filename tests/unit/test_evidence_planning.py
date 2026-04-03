"""Tests for Feature 1: Evidence-Grounded Planning."""

from __future__ import annotations

import pytest
from pydantic_ai import Tool

from forge.agent.loop import PLAN_OVERLAY, create_agent
from forge.agent.tools import ALL_TOOLS, READ_ONLY_TOOLS


class TestReadOnlyTools:
    """Verify READ_ONLY_TOOLS is properly defined."""

    def test_read_only_tools_exist(self):
        assert len(READ_ONLY_TOOLS) > 0

    def test_read_only_tools_are_subset_of_all_tools(self):
        """All read-only tools should have matching names in ALL_TOOLS."""
        all_names = {t.name for t in ALL_TOOLS}
        ro_names = {t.name for t in READ_ONLY_TOOLS}
        assert ro_names.issubset(all_names | {"analyze_impact"})

    def test_read_only_tools_exclude_write_tools(self):
        ro_names = {t.name for t in READ_ONLY_TOOLS}
        assert "write_file" not in ro_names
        assert "edit_file" not in ro_names
        assert "run_command" not in ro_names

    def test_read_only_tools_include_exploration_tools(self):
        ro_names = {t.name for t in READ_ONLY_TOOLS}
        assert "read_file" in ro_names
        assert "search_code" in ro_names
        assert "list_files" in ro_names

    def test_analyze_impact_in_read_only(self):
        ro_names = {t.name for t in READ_ONLY_TOOLS}
        assert "analyze_impact" in ro_names

    def test_read_only_tools_are_tool_instances(self):
        for t in READ_ONLY_TOOLS:
            assert isinstance(t, Tool)


class TestPlanOverlay:
    """Verify PLAN_OVERLAY prompt content."""

    def test_instructs_exploration(self):
        assert "explore" in PLAN_OVERLAY.lower() or "read" in PLAN_OVERLAY.lower()

    def test_instructs_tool_use(self):
        assert "tool" in PLAN_OVERLAY.lower()

    def test_forbids_write_tools(self):
        assert "write_file" in PLAN_OVERLAY or "read-only" in PLAN_OVERLAY.lower()

    def test_requires_evidence(self):
        assert "imagination" in PLAN_OVERLAY.lower() or "evidence" in PLAN_OVERLAY.lower()

    def test_structured_plan_format(self):
        assert "Goal" in PLAN_OVERLAY
        assert "Steps" in PLAN_OVERLAY
        assert "Risks" in PLAN_OVERLAY

    def test_includes_context_section(self):
        """Plan now includes a Context section for discovered files."""
        assert "Context" in PLAN_OVERLAY

    def test_includes_dependencies_section(self):
        """Plan now includes a Dependencies section."""
        assert "Dependencies" in PLAN_OVERLAY
