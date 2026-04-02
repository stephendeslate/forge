"""Tests for embedding generation via Ollama API."""

from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest

from forge.models.embeddings import (
    EMBEDDING_DIM,
    embed_single,
    embed_texts,
    format_embedding_for_pg,
)


# ---------------------------------------------------------------------------
# format_embedding_for_pg
# ---------------------------------------------------------------------------


class TestFormatEmbeddingForPg:
    def test_simple_vector(self):
        result = format_embedding_for_pg([1.0, 2.0, 3.0])
        assert result == "[1.000000,2.000000,3.000000]"

    def test_negative_values(self):
        result = format_embedding_for_pg([-0.5, 0.0, 0.5])
        assert result == "[-0.500000,0.000000,0.500000]"

    def test_empty_vector(self):
        result = format_embedding_for_pg([])
        assert result == "[]"

    def test_precision(self):
        result = format_embedding_for_pg([0.123456789])
        assert result == "[0.123457]"  # 6 decimal places, rounded

    def test_brackets_present(self):
        result = format_embedding_for_pg([1.0])
        assert result.startswith("[")
        assert result.endswith("]")

    def test_no_spaces(self):
        result = format_embedding_for_pg([1.0, 2.0])
        assert " " not in result


# ---------------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------------


def _make_fake_embeddings(n: int) -> list[list[float]]:
    """Generate n fake 768-dim embedding vectors."""
    return [[float(i)] * EMBEDDING_DIM for i in range(n)]


class TestEmbedTexts:
    @pytest.fixture
    def mock_client(self):
        """Patch httpx.AsyncClient to return fake embeddings."""
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        return client

    async def test_single_text(self, mock_client):
        embeddings = _make_fake_embeddings(1)
        response = MagicMock()
        response.json.return_value = {"embeddings": embeddings}
        response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=response)

        with patch("forge.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            result = await embed_texts(["hello world"])

        assert len(result) == 1
        assert len(result[0]) == EMBEDDING_DIM
        mock_client.post.assert_called_once()

    async def test_multiple_texts_single_batch(self, mock_client):
        texts = ["text one", "text two", "text three"]
        embeddings = _make_fake_embeddings(3)
        response = MagicMock()
        response.json.return_value = {"embeddings": embeddings}
        response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=response)

        with patch("forge.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            result = await embed_texts(texts)

        assert len(result) == 3
        # Should be a single batch call (3 < 32)
        assert mock_client.post.call_count == 1

    async def test_batching(self, mock_client):
        texts = [f"text {i}" for i in range(50)]

        # First batch: 32 embeddings, second batch: 18 embeddings
        batch1 = _make_fake_embeddings(32)
        batch2 = _make_fake_embeddings(18)

        resp1 = MagicMock()
        resp1.json.return_value = {"embeddings": batch1}
        resp1.raise_for_status = MagicMock()

        resp2 = MagicMock()
        resp2.json.return_value = {"embeddings": batch2}
        resp2.raise_for_status = MagicMock()

        mock_client.post = AsyncMock(side_effect=[resp1, resp2])

        with patch("forge.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            result = await embed_texts(texts, batch_size=32)

        assert len(result) == 50
        assert mock_client.post.call_count == 2

    async def test_calls_correct_url(self, mock_client):
        embeddings = _make_fake_embeddings(1)
        response = MagicMock()
        response.json.return_value = {"embeddings": embeddings}
        response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=response)

        with (
            patch("forge.models.embeddings.httpx.AsyncClient", return_value=mock_client),
            patch("forge.models.embeddings.settings") as mock_settings,
        ):
            mock_settings.ollama.base_url = "http://localhost:11434"
            mock_settings.ollama.embed_model = "nomic-embed-text-v2-moe"
            await embed_texts(["test"])

        call_args = mock_client.post.call_args
        assert "/api/embed" in call_args[0][0]

    async def test_sends_model_and_input(self, mock_client):
        embeddings = _make_fake_embeddings(1)
        response = MagicMock()
        response.json.return_value = {"embeddings": embeddings}
        response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=response)

        with (
            patch("forge.models.embeddings.httpx.AsyncClient", return_value=mock_client),
            patch("forge.models.embeddings.settings") as mock_settings,
        ):
            mock_settings.ollama.base_url = "http://localhost:11434"
            mock_settings.ollama.embed_model = "test-model"
            await embed_texts(["hello"])

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "test-model"
        assert payload["input"] == ["hello"]

    async def test_http_error_propagates(self, mock_client):
        response = MagicMock()
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        mock_client.post = AsyncMock(return_value=response)

        with patch("forge.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                await embed_texts(["fail"])

    async def test_empty_list(self, mock_client):
        with patch("forge.models.embeddings.httpx.AsyncClient", return_value=mock_client):
            result = await embed_texts([])

        assert result == []
        mock_client.post.assert_not_called()


# ---------------------------------------------------------------------------
# embed_single
# ---------------------------------------------------------------------------


class TestEmbedSingle:
    async def test_delegates_to_embed_texts(self):
        fake_embedding = [0.1] * EMBEDDING_DIM
        with patch("forge.models.embeddings.embed_texts", new_callable=AsyncMock) as mock:
            mock.return_value = [fake_embedding]
            result = await embed_single("query text")

        assert result == fake_embedding
        mock.assert_called_once_with(["query text"])


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_embedding_dim(self):
        assert EMBEDDING_DIM == 768
