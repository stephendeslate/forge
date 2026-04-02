"""Tests for conversation checkpoint DB methods."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.storage.database import Database


@pytest.fixture
def db():
    """Create a Database instance with mocked pool."""
    d = object.__new__(Database)
    d._pool = MagicMock()
    return d


class TestSaveCheckpoint:
    @pytest.mark.asyncio
    async def test_save_checkpoint_calls_execute(self, db):
        db._pool.execute = AsyncMock()
        await db.save_checkpoint(
            "sess-1", "my-cp", '{"messages": []}', '{"tasks": []}', 5,
        )
        db._pool.execute.assert_awaited_once()
        call_args = db._pool.execute.call_args
        assert "INSERT INTO checkpoints" in call_args[0][0]
        assert call_args[0][1] == "sess-1"
        assert call_args[0][2] == "my-cp"
        assert call_args[0][5] == 5

    @pytest.mark.asyncio
    async def test_save_checkpoint_upsert(self, db):
        """Verify the SQL includes ON CONFLICT for upsert behavior."""
        db._pool.execute = AsyncMock()
        await db.save_checkpoint("s1", "cp1", "{}", None, 3)
        sql = db._pool.execute.call_args[0][0]
        assert "ON CONFLICT" in sql
        assert "DO UPDATE" in sql


class TestLoadCheckpoint:
    @pytest.mark.asyncio
    async def test_load_existing(self, db):
        now = datetime.now(timezone.utc)
        row = {
            "agent_history": '{"messages": [1,2,3]}',
            "task_store": '{"tasks": []}',
            "message_count": 3,
            "created_at": now,
        }
        db._pool.fetchrow = AsyncMock(return_value=row)
        result = await db.load_checkpoint("sess-1", "my-cp")
        assert result is not None
        assert result["message_count"] == 3
        assert result["agent_history"] == '{"messages": [1,2,3]}'

    @pytest.mark.asyncio
    async def test_load_missing(self, db):
        db._pool.fetchrow = AsyncMock(return_value=None)
        result = await db.load_checkpoint("sess-1", "nope")
        assert result is None


class TestListCheckpoints:
    @pytest.mark.asyncio
    async def test_list_empty(self, db):
        db._pool.fetch = AsyncMock(return_value=[])
        result = await db.list_checkpoints("sess-1")
        assert result == []

    @pytest.mark.asyncio
    async def test_list_populated(self, db):
        now = datetime.now(timezone.utc)
        rows = [
            {"name": "cp-2", "message_count": 10, "created_at": now},
            {"name": "cp-1", "message_count": 5, "created_at": now},
        ]
        db._pool.fetch = AsyncMock(return_value=rows)
        result = await db.list_checkpoints("sess-1")
        assert len(result) == 2
        assert result[0]["name"] == "cp-2"


class TestDeleteCheckpoint:
    @pytest.mark.asyncio
    async def test_delete_existing(self, db):
        db._pool.execute = AsyncMock(return_value="DELETE 1")
        result = await db.delete_checkpoint("sess-1", "cp-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_missing(self, db):
        db._pool.execute = AsyncMock(return_value="DELETE 0")
        result = await db.delete_checkpoint("sess-1", "nope")
        assert result is False
