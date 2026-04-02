"""Tests for memory tools — mock DB, test save/recall/tool functions."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.permissions import PermissionPolicy
from forge.agent.tools import save_memory, recall_memories


@pytest.fixture
def ctx(tmp_path):
    mock = MagicMock()
    mock.deps = AgentDeps(
        cwd=tmp_path,
        console=Console(file=None, force_terminal=False, no_color=True),
        permission=PermissionPolicy.YOLO,
    )
    return mock


class TestSaveMemoryTool:
    async def test_unavailable_when_no_db(self, ctx):
        ctx.deps.memory_db = None
        ctx.deps.memory_project = None
        result = await save_memory(ctx, "feedback", "test", "content")
        assert "unavailable" in result.lower()

    async def test_invalid_category(self, ctx):
        ctx.deps.memory_db = MagicMock()
        ctx.deps.memory_project = "test"
        result = await save_memory(ctx, "invalid_cat", "test", "content")
        assert "Invalid category" in result

    async def test_save_success(self, ctx):
        ctx.deps.memory_db = MagicMock()
        ctx.deps.memory_project = "test"

        mock_save = AsyncMock(return_value=42)
        with patch("forge.agent.memory.save_memory_to_db", mock_save):
            result = await save_memory(ctx, "feedback", "test subject", "test content")
        assert "id=42" in result
        assert "feedback" in result


class TestRecallMemoriesTool:
    async def test_unavailable_when_no_db(self, ctx):
        ctx.deps.memory_db = None
        ctx.deps.memory_project = None
        result = await recall_memories(ctx, "test query")
        assert "unavailable" in result.lower()

    async def test_no_results(self, ctx):
        ctx.deps.memory_db = MagicMock()
        ctx.deps.memory_project = "test"

        mock_recall = AsyncMock(return_value=[])
        with patch("forge.agent.memory.recall_from_db", mock_recall):
            result = await recall_memories(ctx, "test query")
        assert "No memories found" in result

    async def test_returns_formatted_results(self, ctx):
        from forge.storage.database import MemoryRow

        ctx.deps.memory_db = MagicMock()
        ctx.deps.memory_project = "test"

        mock_rows = [
            MemoryRow(
                id=1, project="test", category="feedback",
                subject="use tabs", content="User prefers tabs over spaces",
                score=0.85,
            ),
            MemoryRow(
                id=2, project="test", category="project",
                subject="auth rewrite", content="Auth middleware being rewritten for compliance",
                score=0.72,
            ),
        ]

        mock_recall = AsyncMock(return_value=mock_rows)
        with patch("forge.agent.memory.recall_from_db", mock_recall):
            result = await recall_memories(ctx, "preferences")
        assert "Found 2 memories" in result
        assert "use tabs" in result
        assert "auth rewrite" in result
        assert "0.85" in result


class TestMemoryModule:
    """Test the memory.py high-level functions."""

    async def test_save_memory_to_db(self):
        from forge.agent.memory import save_memory_to_db

        mock_db = AsyncMock()
        mock_db.save_memory = AsyncMock(return_value=7)
        mock_db.count_memories = AsyncMock(return_value=10)

        with patch("forge.agent.memory.embed_single", new_callable=AsyncMock, return_value=[0.1] * 768):
            with patch("forge.agent.memory.format_embedding_for_pg", return_value="[0.1,...]"):
                mid = await save_memory_to_db(mock_db, "proj", "feedback", "subj", "content")

        assert mid == 7
        mock_db.save_memory.assert_called_once()
        mock_db.count_memories.assert_called_once_with("proj")
        mock_db.prune_memories.assert_not_called()

    async def test_save_auto_prunes(self):
        from forge.agent.memory import save_memory_to_db

        mock_db = AsyncMock()
        mock_db.save_memory = AsyncMock(return_value=51)
        mock_db.count_memories = AsyncMock(return_value=55)
        mock_db.prune_memories = AsyncMock(return_value=5)

        with patch("forge.agent.memory.embed_single", new_callable=AsyncMock, return_value=[0.1] * 768):
            with patch("forge.agent.memory.format_embedding_for_pg", return_value="[0.1,...]"):
                await save_memory_to_db(mock_db, "proj", "feedback", "subj", "content")

        mock_db.prune_memories.assert_called_once_with("proj", keep=50)

    async def test_recall_from_db(self):
        from forge.agent.memory import recall_from_db
        from forge.storage.database import MemoryRow

        mock_rows = [MemoryRow(id=1, project="p", category="user", subject="s", content="c", score=0.9)]
        mock_db = AsyncMock()
        mock_db.search_memories = AsyncMock(return_value=mock_rows)

        with patch("forge.agent.memory.embed_single", new_callable=AsyncMock, return_value=[0.2] * 768):
            with patch("forge.agent.memory.format_embedding_for_pg", return_value="[0.2,...]"):
                results = await recall_from_db(mock_db, "proj", "query")

        assert len(results) == 1
        assert results[0].subject == "s"

    async def test_get_startup_memories(self):
        from forge.agent.memory import get_startup_memories
        from forge.storage.database import MemoryRow

        mock_db = AsyncMock()
        mock_db.list_memories = AsyncMock(return_value=[
            MemoryRow(id=1, project="p", category="user", subject="role", content="engineer"),
        ])

        results = await get_startup_memories(mock_db, "proj", limit=5)
        assert len(results) == 1
        mock_db.list_memories.assert_called_once_with("proj", limit=5)
