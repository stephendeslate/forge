"""Tests for TaskStore — CRUD, serialization, dependencies."""

import json

import pytest

from forge.agent.task_store import Task, TaskStatus, TaskStore


@pytest.fixture
def store():
    return TaskStore()


class TestTaskStoreCreate:
    def test_create_auto_increments_id(self, store):
        t1 = store.create("First", "desc 1")
        t2 = store.create("Second", "desc 2")
        assert t1.id == "t1"
        assert t2.id == "t2"

    def test_create_sets_defaults(self, store):
        t = store.create("Test", "description")
        assert t.status == TaskStatus.PENDING
        assert t.blocked_by == []
        assert t.blocks == []
        assert t.active_form is None

    def test_create_with_active_form(self, store):
        t = store.create("Build app", "build it", active_form="Building app")
        assert t.active_form == "Building app"


class TestTaskStoreGet:
    def test_get_existing(self, store):
        created = store.create("Test", "desc")
        got = store.get(created.id)
        assert got is created

    def test_get_missing(self, store):
        assert store.get("t999") is None


class TestTaskStoreUpdate:
    def test_update_status(self, store):
        t = store.create("Test", "desc")
        updated = store.update(t.id, status=TaskStatus.IN_PROGRESS)
        assert updated.status == TaskStatus.IN_PROGRESS

    def test_update_subject(self, store):
        t = store.create("Old", "desc")
        store.update(t.id, subject="New")
        assert t.subject == "New"

    def test_update_missing_returns_none(self, store):
        assert store.update("t999", subject="x") is None

    def test_add_blocked_by_sets_reverse(self, store):
        t1 = store.create("First", "desc")
        t2 = store.create("Second", "desc")
        store.update(t2.id, add_blocked_by=[t1.id])
        assert t1.id in t2.blocked_by
        assert t2.id in t1.blocks

    def test_add_blocks_sets_reverse(self, store):
        t1 = store.create("First", "desc")
        t2 = store.create("Second", "desc")
        store.update(t1.id, add_blocks=[t2.id])
        assert t2.id in t1.blocks
        assert t1.id in t2.blocked_by


class TestTaskStoreListing:
    def test_list_all_excludes_deleted(self, store):
        t1 = store.create("Active", "desc")
        t2 = store.create("Deleted", "desc")
        store.update(t2.id, status=TaskStatus.DELETED)
        result = store.list_all()
        assert len(result) == 1
        assert result[0].id == t1.id

    def test_list_open(self, store):
        t1 = store.create("Pending", "desc")
        t2 = store.create("Done", "desc")
        store.update(t2.id, status=TaskStatus.COMPLETED)
        result = store.list_open()
        assert len(result) == 1
        assert result[0].id == t1.id

    def test_get_active(self, store):
        store.create("Pending", "desc")
        t2 = store.create("Active", "desc")
        store.update(t2.id, status=TaskStatus.IN_PROGRESS)
        active = store.get_active()
        assert active.id == t2.id

    def test_get_active_none(self, store):
        store.create("Pending", "desc")
        assert store.get_active() is None


class TestTaskStorePrompt:
    def test_to_prompt_empty(self, store):
        assert store.to_prompt() == ""

    def test_to_prompt_with_tasks(self, store):
        store.create("Fix bug", "fix it")
        t2 = store.create("Write test", "write tests")
        store.update(t2.id, status=TaskStatus.IN_PROGRESS)
        prompt = store.to_prompt()
        assert "[ ] t1:" in prompt
        assert "[~] t2:" in prompt

    def test_to_prompt_shows_active_form(self, store):
        t = store.create("Build", "build it", active_form="Building app")
        store.update(t.id, status=TaskStatus.IN_PROGRESS)
        prompt = store.to_prompt()
        assert "Building app" in prompt

    def test_to_prompt_shows_blocked(self, store):
        t1 = store.create("First", "desc")
        t2 = store.create("Second", "desc")
        store.update(t2.id, add_blocked_by=[t1.id])
        prompt = store.to_prompt()
        assert "blocked by t1" in prompt


class TestTaskStoreSerialization:
    def test_round_trip(self, store):
        t1 = store.create("First", "desc 1", active_form="Doing first")
        t2 = store.create("Second", "desc 2")
        store.update(t1.id, status=TaskStatus.IN_PROGRESS)
        store.update(t2.id, add_blocked_by=[t1.id])

        json_str = store.to_json()
        restored = TaskStore.from_json(json_str)

        assert restored._counter == 2
        r1 = restored.get("t1")
        r2 = restored.get("t2")
        assert r1.subject == "First"
        assert r1.status == TaskStatus.IN_PROGRESS
        assert r1.active_form == "Doing first"
        assert r2.blocked_by == ["t1"]
        assert r1.blocks == ["t2"]

    def test_to_json_is_valid_json(self, store):
        store.create("Test", "desc")
        data = json.loads(store.to_json())
        assert "counter" in data
        assert "tasks" in data

    def test_from_json_empty(self):
        store = TaskStore.from_json('{"counter": 0, "tasks": []}')
        assert store.list_all() == []
