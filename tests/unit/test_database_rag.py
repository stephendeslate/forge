"""Tests for Database RAG operations (chunk storage and search).

These tests mock asyncpg to test Database methods without a real PostgreSQL instance.
"""

from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from forge.storage.database import ChunkRow, Database


# ---------------------------------------------------------------------------
# ChunkRow dataclass
# ---------------------------------------------------------------------------


class TestChunkRow:
    def test_construction(self):
        row = ChunkRow(
            id=1,
            project="test",
            file_path="main.py",
            chunk_type="function_definition",
            name="main",
            content="def main(): pass",
            start_line=1,
            end_line=3,
            token_count=10,
            file_hash="abc",
        )
        assert row.id == 1
        assert row.score == 0.0  # default

    def test_score_default(self):
        row = ChunkRow(
            id=1, project="p", file_path="f", chunk_type="t",
            name=None, content="c", start_line=1, end_line=1,
            token_count=1, file_hash="h",
        )
        assert row.score == 0.0

    def test_score_custom(self):
        row = ChunkRow(
            id=1, project="p", file_path="f", chunk_type="t",
            name=None, content="c", start_line=1, end_line=1,
            token_count=1, file_hash="h", score=0.95,
        )
        assert row.score == 0.95

    def test_name_optional(self):
        row = ChunkRow(
            id=1, project="p", file_path="f", chunk_type="t",
            name=None, content="c", start_line=1, end_line=1,
            token_count=1, file_hash="h",
        )
        assert row.name is None


# ---------------------------------------------------------------------------
# Database connection lifecycle
# ---------------------------------------------------------------------------


class TestDatabaseConnection:
    def test_pool_raises_when_not_connected(self):
        db = Database()
        with pytest.raises(RuntimeError, match="not connected"):
            _ = db.pool

    async def test_connect_creates_pool(self):
        db = Database(dsn="postgresql://test@localhost/test")
        mock_pool = AsyncMock()
        with patch("forge.storage.database.asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pool):
            await db.connect()
        assert db._pool is mock_pool

    async def test_close_clears_pool(self):
        db = Database()
        mock_pool = AsyncMock()
        db._pool = mock_pool
        await db.close()
        assert db._pool is None
        mock_pool.close.assert_called_once()

    async def test_close_when_not_connected(self):
        db = Database()
        await db.close()  # should not raise


# ---------------------------------------------------------------------------
# get_file_hash
# ---------------------------------------------------------------------------


class TestGetFileHash:
    @pytest.fixture
    def db(self):
        db = Database()
        db._pool = AsyncMock()
        return db

    async def test_returns_hash_when_found(self, db):
        db._pool.fetchrow = AsyncMock(return_value={"file_hash": "abc123"})
        result = await db.get_file_hash("project", "file.py")
        assert result == "abc123"

    async def test_returns_none_when_not_found(self, db):
        db._pool.fetchrow = AsyncMock(return_value=None)
        result = await db.get_file_hash("project", "missing.py")
        assert result is None

    async def test_queries_correct_project_and_path(self, db):
        db._pool.fetchrow = AsyncMock(return_value=None)
        await db.get_file_hash("myproj", "src/main.py")
        call_args = db._pool.fetchrow.call_args[0]
        assert "myproj" in call_args
        assert "src/main.py" in call_args


# ---------------------------------------------------------------------------
# delete_file_chunks
# ---------------------------------------------------------------------------


class TestDeleteFileChunks:
    @pytest.fixture
    def db(self):
        db = Database()
        db._pool = AsyncMock()
        return db

    async def test_returns_count(self, db):
        db._pool.execute = AsyncMock(return_value="DELETE 5")
        result = await db.delete_file_chunks("proj", "old.py")
        assert result == 5

    async def test_returns_zero(self, db):
        db._pool.execute = AsyncMock(return_value="DELETE 0")
        result = await db.delete_file_chunks("proj", "nonexistent.py")
        assert result == 0


# ---------------------------------------------------------------------------
# insert_chunks
# ---------------------------------------------------------------------------


class TestInsertChunks:
    @pytest.fixture
    def db(self):
        db = Database()
        db._pool = AsyncMock()
        return db

    async def test_empty_list_returns_zero(self, db):
        result = await db.insert_chunks([])
        assert result == 0
        db._pool.executemany.assert_not_called()

    async def test_inserts_records(self, db):
        chunks = [
            {
                "project": "test",
                "file_path": "main.py",
                "chunk_type": "function_definition",
                "name": "main",
                "content": "def main(): pass",
                "start_line": 1,
                "end_line": 3,
                "token_count": 10,
                "embedding": "[0.1,0.2,0.3]",
                "file_hash": "abc123",
            }
        ]
        db._pool.executemany = AsyncMock()
        result = await db.insert_chunks(chunks)
        assert result == 1
        db._pool.executemany.assert_called_once()

    async def test_returns_count_of_records(self, db):
        chunks = [
            {
                "project": "test",
                "file_path": f"file{i}.py",
                "chunk_type": "block",
                "name": None,
                "content": f"content {i}",
                "start_line": 1,
                "end_line": 1,
                "token_count": 5,
                "embedding": "[0.1]",
                "file_hash": f"hash{i}",
            }
            for i in range(5)
        ]
        db._pool.executemany = AsyncMock()
        result = await db.insert_chunks(chunks)
        assert result == 5

    async def test_record_tuple_order(self, db):
        chunk = {
            "project": "proj",
            "file_path": "f.py",
            "chunk_type": "block",
            "name": "func",
            "content": "code",
            "start_line": 10,
            "end_line": 20,
            "token_count": 50,
            "embedding": "[0.1]",
            "file_hash": "h",
        }
        db._pool.executemany = AsyncMock()
        await db.insert_chunks([chunk])

        records = db._pool.executemany.call_args[0][1]
        assert records[0] == ("proj", "f.py", "block", "func", "code", 10, 20, 50, "[0.1]", "h")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    @pytest.fixture
    def db(self):
        db = Database()
        db._pool = AsyncMock()
        return db

    def _make_row(self, id: int = 1, score: float = 0.9) -> dict:
        return {
            "id": id,
            "project": "test",
            "file_path": "main.py",
            "chunk_type": "function_definition",
            "name": "main",
            "content": "def main(): pass",
            "start_line": 1,
            "end_line": 5,
            "token_count": 10,
            "file_hash": "abc",
            "score": score,
        }

    async def test_returns_chunk_rows(self, db):
        db._pool.fetch = AsyncMock(return_value=[self._make_row()])
        results = await db.search("[0.1]", "test", limit=10, min_score=0.3)
        assert len(results) == 1
        assert isinstance(results[0], ChunkRow)
        assert results[0].score == 0.9

    async def test_empty_results(self, db):
        db._pool.fetch = AsyncMock(return_value=[])
        results = await db.search("[0.1]", "test")
        assert results == []

    async def test_multiple_results(self, db):
        rows = [self._make_row(id=1, score=0.9), self._make_row(id=2, score=0.7)]
        db._pool.fetch = AsyncMock(return_value=rows)
        results = await db.search("[0.1]", "test")
        assert len(results) == 2
        assert results[0].score == 0.9
        assert results[1].score == 0.7

    async def test_passes_parameters(self, db):
        db._pool.fetch = AsyncMock(return_value=[])
        await db.search("[0.1,0.2]", "myproject", limit=5, min_score=0.5)
        call_args = db._pool.fetch.call_args[0]
        # positional args after the SQL: embedding, project, limit, min_score
        assert "[0.1,0.2]" in call_args
        assert "myproject" in call_args
        assert 5 in call_args
        assert 0.5 in call_args


# ---------------------------------------------------------------------------
# get_project_stats
# ---------------------------------------------------------------------------


class TestGetProjectStats:
    @pytest.fixture
    def db(self):
        db = Database()
        db._pool = AsyncMock()
        return db

    async def test_returns_stats(self, db):
        db._pool.fetchrow = AsyncMock(return_value={
            "chunk_count": 42,
            "file_count": 10,
            "last_indexed": "2024-01-01T00:00:00",
        })
        stats = await db.get_project_stats("myproject")
        assert stats["chunk_count"] == 42
        assert stats["file_count"] == 10

    async def test_returns_none_row_fallback(self, db):
        db._pool.fetchrow = AsyncMock(return_value=None)
        stats = await db.get_project_stats("empty")
        assert stats["chunk_count"] == 0
