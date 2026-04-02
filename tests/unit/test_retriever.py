"""Tests for RAG retrieval and context formatting."""

from unittest.mock import AsyncMock, patch

import pytest

from forge.rag.retriever import format_context, retrieve
from forge.storage.database import ChunkRow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunk(
    id: int = 1,
    file_path: str = "src/main.py",
    name: str | None = "main",
    content: str = "def main(): pass",
    start_line: int = 1,
    end_line: int = 5,
    token_count: int = 50,
    score: float = 0.85,
) -> ChunkRow:
    return ChunkRow(
        id=id,
        project="test",
        file_path=file_path,
        chunk_type="function_definition",
        name=name,
        content=content,
        start_line=start_line,
        end_line=end_line,
        token_count=token_count,
        file_hash="abc123",
        score=score,
    )


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------


class TestFormatContext:
    def test_empty_chunks(self):
        assert format_context([]) == ""

    def test_single_chunk_structure(self):
        chunk = _make_chunk()
        result = format_context([chunk])
        assert result.startswith("<context>")
        assert result.endswith("</context>")
        assert "src/main.py" in result
        assert "main" in result
        assert "lines 1-5" in result
        assert "score: 0.85" in result
        assert "def main(): pass" in result

    def test_chunk_without_name(self):
        chunk = _make_chunk(name=None)
        result = format_context([chunk])
        assert "src/main.py" in result
        # Should not have " — None"
        assert "None" not in result

    def test_multiple_chunks(self):
        chunks = [
            _make_chunk(id=1, file_path="a.py", name="func_a", score=0.9),
            _make_chunk(id=2, file_path="b.py", name="func_b", score=0.7),
        ]
        result = format_context(chunks)
        assert "a.py" in result
        assert "b.py" in result
        assert "func_a" in result
        assert "func_b" in result

    def test_code_fences(self):
        chunk = _make_chunk(content="x = 42")
        result = format_context([chunk])
        assert "```\nx = 42\n```" in result

    def test_score_formatting(self):
        chunk = _make_chunk(score=0.123456)
        result = format_context([chunk])
        assert "score: 0.12" in result  # 2 decimal places

    def test_header_format_with_name(self):
        chunk = _make_chunk(file_path="lib/utils.py", name="helper", start_line=10, end_line=20, score=0.75)
        result = format_context([chunk])
        assert "# lib/utils.py — helper (lines 10-20, score: 0.75)" in result

    def test_header_format_without_name(self):
        chunk = _make_chunk(file_path="lib/utils.py", name=None, start_line=10, end_line=20, score=0.75)
        result = format_context([chunk])
        assert "# lib/utils.py (lines 10-20, score: 0.75)" in result


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------


class TestRetrieve:
    @pytest.fixture
    def mock_db(self):
        return AsyncMock()

    async def test_basic_retrieval(self, mock_db):
        chunks = [_make_chunk(token_count=100)]
        mock_db.search = AsyncMock(return_value=chunks)

        with patch("forge.rag.retriever.embed_single", new_callable=AsyncMock, return_value=[0.1] * 768):
            with patch("forge.rag.retriever.format_embedding_for_pg", return_value="[0.1]"):
                result = await retrieve("test query", "myproject", mock_db)

        assert len(result) == 1
        assert result[0].file_path == "src/main.py"

    async def test_passes_parameters_to_search(self, mock_db):
        mock_db.search = AsyncMock(return_value=[])

        with patch("forge.rag.retriever.embed_single", new_callable=AsyncMock, return_value=[0.1] * 768):
            with patch("forge.rag.retriever.format_embedding_for_pg", return_value="[0.1]"):
                await retrieve("query", "proj", mock_db, limit=5, min_score=0.5)

        mock_db.search.assert_called_once()
        call_kwargs = mock_db.search.call_args
        assert call_kwargs.kwargs["limit"] == 5
        assert call_kwargs.kwargs["min_score"] == 0.5

    async def test_token_budget_enforcement(self, mock_db):
        # Three chunks of 1500 tokens each — budget is 3000
        chunks = [
            _make_chunk(id=1, token_count=1500, score=0.9),
            _make_chunk(id=2, token_count=1500, score=0.8),
            _make_chunk(id=3, token_count=1500, score=0.7),
        ]
        mock_db.search = AsyncMock(return_value=chunks)

        with patch("forge.rag.retriever.embed_single", new_callable=AsyncMock, return_value=[0.1] * 768):
            with patch("forge.rag.retriever.format_embedding_for_pg", return_value="[0.1]"):
                result = await retrieve("query", "proj", mock_db, max_tokens=3000)

        # Only first two fit within 3000 token budget
        assert len(result) == 2

    async def test_token_budget_exact_fit(self, mock_db):
        chunks = [
            _make_chunk(id=1, token_count=1000),
            _make_chunk(id=2, token_count=1000),
            _make_chunk(id=3, token_count=1000),
        ]
        mock_db.search = AsyncMock(return_value=chunks)

        with patch("forge.rag.retriever.embed_single", new_callable=AsyncMock, return_value=[0.1] * 768):
            with patch("forge.rag.retriever.format_embedding_for_pg", return_value="[0.1]"):
                result = await retrieve("query", "proj", mock_db, max_tokens=3000)

        assert len(result) == 3  # exactly fits

    async def test_token_budget_single_oversized_chunk(self, mock_db):
        # Single chunk exceeds budget
        chunks = [_make_chunk(token_count=5000)]
        mock_db.search = AsyncMock(return_value=chunks)

        with patch("forge.rag.retriever.embed_single", new_callable=AsyncMock, return_value=[0.1] * 768):
            with patch("forge.rag.retriever.format_embedding_for_pg", return_value="[0.1]"):
                result = await retrieve("query", "proj", mock_db, max_tokens=3000)

        assert len(result) == 0

    async def test_no_results(self, mock_db):
        mock_db.search = AsyncMock(return_value=[])

        with patch("forge.rag.retriever.embed_single", new_callable=AsyncMock, return_value=[0.1] * 768):
            with patch("forge.rag.retriever.format_embedding_for_pg", return_value="[0.1]"):
                result = await retrieve("obscure query", "proj", mock_db)

        assert result == []

    async def test_embeds_query(self, mock_db):
        mock_db.search = AsyncMock(return_value=[])

        with patch("forge.rag.retriever.embed_single", new_callable=AsyncMock, return_value=[0.1] * 768) as mock_embed:
            with patch("forge.rag.retriever.format_embedding_for_pg", return_value="[0.1]"):
                await retrieve("find the router", "proj", mock_db)

        mock_embed.assert_called_once_with("find the router")

    async def test_preserves_result_order(self, mock_db):
        chunks = [
            _make_chunk(id=1, name="best", score=0.95, token_count=100),
            _make_chunk(id=2, name="good", score=0.80, token_count=100),
            _make_chunk(id=3, name="ok", score=0.60, token_count=100),
        ]
        mock_db.search = AsyncMock(return_value=chunks)

        with patch("forge.rag.retriever.embed_single", new_callable=AsyncMock, return_value=[0.1] * 768):
            with patch("forge.rag.retriever.format_embedding_for_pg", return_value="[0.1]"):
                result = await retrieve("query", "proj", mock_db)

        assert [r.name for r in result] == ["best", "good", "ok"]

    async def test_default_parameters(self, mock_db):
        mock_db.search = AsyncMock(return_value=[])

        with patch("forge.rag.retriever.embed_single", new_callable=AsyncMock, return_value=[0.1] * 768):
            with patch("forge.rag.retriever.format_embedding_for_pg", return_value="[0.1]"):
                await retrieve("query", "proj", mock_db)

        call_kwargs = mock_db.search.call_args
        assert call_kwargs.kwargs["limit"] == 8
        assert call_kwargs.kwargs["min_score"] == 0.3
