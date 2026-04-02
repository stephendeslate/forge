"""Tests for RAG auto-indexing (staleness detection + targeted reindex)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.rag.indexer import find_stale_files, reindex_files


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.get_project_stats = AsyncMock()
    db.get_file_hash = AsyncMock(return_value=None)
    db.delete_file_chunks = AsyncMock()
    db.insert_chunks = AsyncMock(return_value=0)
    return db


class TestFindStaleFiles:
    @pytest.mark.asyncio
    async def test_never_indexed_returns_empty(self, mock_db, tmp_path):
        """If project was never indexed, don't auto-index."""
        mock_db.get_project_stats.return_value = {
            "chunk_count": 0, "file_count": 0, "last_indexed": None,
        }
        (tmp_path / "test.py").write_text("x = 1")
        result = await find_stale_files(tmp_path, mock_db, "proj")
        assert result == []

    @pytest.mark.asyncio
    async def test_none_stale(self, mock_db, tmp_path):
        """All files older than last_indexed."""
        f = tmp_path / "test.py"
        f.write_text("x = 1")
        # Set last_indexed to the future
        mock_db.get_project_stats.return_value = {
            "chunk_count": 5, "file_count": 1,
            "last_indexed": datetime(2099, 1, 1, tzinfo=timezone.utc),
        }
        result = await find_stale_files(tmp_path, mock_db, "proj")
        assert result == []

    @pytest.mark.asyncio
    async def test_with_changes(self, mock_db, tmp_path):
        """File touched after last_indexed should be stale."""
        f = tmp_path / "test.py"
        f.write_text("x = 1")
        # Set last_indexed to the past
        mock_db.get_project_stats.return_value = {
            "chunk_count": 5, "file_count": 1,
            "last_indexed": datetime(2000, 1, 1, tzinfo=timezone.utc),
        }
        result = await find_stale_files(tmp_path, mock_db, "proj")
        assert len(result) == 1
        assert result[0].name == "test.py"


class TestReindexFiles:
    @pytest.mark.asyncio
    async def test_skips_unchanged(self, mock_db, tmp_path):
        """If file hash matches stored hash, skip re-embed."""
        f = tmp_path / "test.py"
        f.write_text("x = 1")

        # Simulate hash match
        import hashlib
        fhash = hashlib.sha256("x = 1".encode()).hexdigest()
        mock_db.get_file_hash.return_value = fhash

        with patch("forge.rag.indexer.embed_texts", new_callable=AsyncMock) as mock_embed:
            stats = await reindex_files([f], tmp_path, mock_db, "proj")

        assert stats["files_skipped"] == 1
        assert stats["files_indexed"] == 0
        mock_embed.assert_not_awaited() if hasattr(mock_embed, "assert_not_awaited") else None

    @pytest.mark.asyncio
    async def test_reindexes_changed(self, mock_db, tmp_path):
        """Changed file (hash mismatch) should be reindexed."""
        f = tmp_path / "test.py"
        f.write_text("def hello():\n    pass\n")
        mock_db.get_file_hash.return_value = "old-hash-doesnt-match"
        mock_db.insert_chunks.return_value = 1

        with patch("forge.rag.indexer.embed_texts", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]
            stats = await reindex_files([f], tmp_path, mock_db, "proj")

        assert stats["files_indexed"] == 1
        assert stats["chunks_stored"] >= 1
        mock_db.delete_file_chunks.assert_awaited()

    @pytest.mark.asyncio
    async def test_handles_missing_file(self, mock_db, tmp_path):
        """Missing file should be skipped without error."""
        missing = tmp_path / "gone.py"
        stats = await reindex_files([missing], tmp_path, mock_db, "proj")
        assert stats["files_skipped"] == 1
        assert stats["files_indexed"] == 0
