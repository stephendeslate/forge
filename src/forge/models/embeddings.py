"""Embedding generation via nomic-embed-text-v2-moe through Ollama /api/embed."""

from __future__ import annotations

import httpx

from forge.config import settings

# nomic-embed-text-v2-moe produces 768-dim embeddings
EMBEDDING_DIM = 768


async def embed_texts(texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
    """Embed a list of texts using the configured embedding model.

    Returns list of 768-dim float vectors.
    Batches requests to avoid overwhelming Ollama.
    """
    all_embeddings: list[list[float]] = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = await client.post(
                f"{settings.ollama.base_url}/api/embed",
                json={
                    "model": settings.ollama.embed_model,
                    "input": batch,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data["embeddings"]
            all_embeddings.extend(embeddings)

    return all_embeddings


async def embed_single(text: str) -> list[float]:
    """Embed a single text. Convenience wrapper."""
    results = await embed_texts([text])
    return results[0]


def format_embedding_for_pg(embedding: list[float]) -> str:
    """Format an embedding vector as a pgvector vector string literal."""
    return "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
