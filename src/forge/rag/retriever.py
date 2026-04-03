"""Query pgvector + BM25 full-text + Reciprocal Rank Fusion for RAG injection."""

from __future__ import annotations

import asyncio

from forge.models.embeddings import embed_single, format_embedding_for_pg
from forge.storage.database import ChunkRow, Database


async def retrieve(
    query: str,
    project: str,
    db: Database,
    *,
    limit: int = 20,
    min_score: float = 0.3,
    max_tokens: int = 0,
) -> list[ChunkRow]:
    """Hybrid retrieval: vector similarity + BM25 full-text, fused with RRF.

    Runs both searches in parallel, combines results using Reciprocal Rank
    Fusion (k=60), and enforces a token budget on the final selection.
    """
    if max_tokens <= 0:
        from forge.config import settings
        max_tokens = settings.agent.rag_max_tokens

    query_embedding = await embed_single(query)
    embedding_str = format_embedding_for_pg(query_embedding)

    # Run both searches in parallel
    vector_results, text_results = await asyncio.gather(
        db.search(embedding_str, project, limit=limit, min_score=min_score),
        db.text_search(query, project, limit=limit),
    )

    # Reciprocal Rank Fusion (k=60 is standard)
    k = 60
    rrf_scores: dict[int, float] = {}
    chunk_map: dict[int, ChunkRow] = {}

    for rank, chunk in enumerate(vector_results):
        rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0) + 1.0 / (k + rank + 1)
        chunk_map[chunk.id] = chunk

    for rank, chunk in enumerate(text_results):
        rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0) + 1.0 / (k + rank + 1)
        chunk_map[chunk.id] = chunk

    # Sort by fused score, descending
    ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    # Enforce token budget
    selected: list[ChunkRow] = []
    total_tokens = 0
    for chunk_id in ranked_ids[:limit]:
        chunk = chunk_map[chunk_id]
        if total_tokens + chunk.token_count > max_tokens:
            break
        chunk.score = rrf_scores[chunk_id]
        selected.append(chunk)
        total_tokens += chunk.token_count

    return selected


def format_context(chunks: list[ChunkRow]) -> str:
    """Format retrieved chunks as context for prompt injection."""
    if not chunks:
        return ""

    sections: list[str] = []
    for chunk in chunks:
        header = f"# {chunk.file_path}"
        if chunk.name:
            header += f" — {chunk.name}"
        header += f" (lines {chunk.start_line}-{chunk.end_line}, score: {chunk.score:.2f})"

        sections.append(f"{header}\n```\n{chunk.content}\n```")

    return "<context>\n" + "\n\n".join(sections) + "\n</context>"
