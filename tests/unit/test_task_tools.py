"""Tests for task agent tool functions."""

import pytest
from unittest.mock import MagicMock

from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.permissions import PermissionPolicy
from forge.agent.task_store import TaskStore
from forge.agent.tools import task_create, task_update, task_list, task_get


@pytest.fixture
def ctx(tmp_path):
    mock = MagicMock()
    mock.deps = AgentDeps(
        cwd=tmp_path,
        console=Console(file=None, force_terminal=False, no_color=True),
        permission=PermissionPolicy.YOLO,
    )
    mock.deps.task_store = TaskStore()
    return mock


class TestTaskCreate:
    async def test_create_task(self, ctx):
        result = await task_create(ctx, "Fix bug", "Fix the login bug")
        assert "t1" in result
        assert "Fix bug" in result

    async def test_create_unavailable(self, ctx):
        ctx.deps.task_store = None
        result = await task_create(ctx, "Test", "desc")
        assert "unavailable" in result.lower()

    async def test_create_with_active_form(self, ctx):
        result = await task_create(ctx, "Build", "build it", active_form="Building")
        assert "t1" in result
        task = ctx.deps.task_store.get("t1")
        assert task.active_form == "Building"


class TestTaskUpdate:
    async def test_update_status(self, ctx):
        await task_create(ctx, "Test", "desc")
        result = await task_update(ctx, "t1", status="in_progress")
        assert "in_progress" in result

    async def test_update_invalid_status(self, ctx):
        await task_create(ctx, "Test", "desc")
        result = await task_update(ctx, "t1", status="invalid")
        assert "Invalid status" in result

    async def test_update_missing_task(self, ctx):
        result = await task_update(ctx, "t999", status="completed")
        assert "not found" in result

    async def test_update_unavailable(self, ctx):
        ctx.deps.task_store = None
        result = await task_update(ctx, "t1", status="completed")
        assert "unavailable" in result.lower()

    async def test_update_blocked_by(self, ctx):
        await task_create(ctx, "First", "desc")
        await task_create(ctx, "Second", "desc")
        result = await task_update(ctx, "t2", add_blocked_by="t1")
        assert "t2" in result
        t2 = ctx.deps.task_store.get("t2")
        assert "t1" in t2.blocked_by


class TestTaskList:
    async def test_list_empty(self, ctx):
        result = await task_list(ctx)
        assert "No tasks" in result

    async def test_list_tasks(self, ctx):
        await task_create(ctx, "First", "desc")
        await task_create(ctx, "Second", "desc")
        result = await task_list(ctx)
        assert "t1" in result
        assert "t2" in result
        assert "First" in result

    async def test_list_unavailable(self, ctx):
        ctx.deps.task_store = None
        result = await task_list(ctx)
        assert "unavailable" in result.lower()


class TestTaskGet:
    async def test_get_task(self, ctx):
        await task_create(ctx, "Test task", "detailed description")
        result = await task_get(ctx, "t1")
        assert "Test task" in result
        assert "detailed description" in result
        assert "pending" in result

    async def test_get_missing(self, ctx):
        result = await task_get(ctx, "t999")
        assert "not found" in result

    async def test_get_unavailable(self, ctx):
        ctx.deps.task_store = None
        result = await task_get(ctx, "t1")
        assert "unavailable" in result.lower()
