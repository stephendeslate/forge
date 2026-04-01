"""Query pgvector + rank + format context for RAG injection."""

from __future__ import annotations

from forge.models.embeddings import embed_single, format_embedding_for_pg
from forge.storage.database import ChunkRow, Database


async def retrieve(
    query: str,
    project: str,
    db: Database,
    *,
    limit: int = 8,
    min_score: float = 0.3,
    max_tokens: int = 3000,
) -> list[ChunkRow]:
    """Retrieve relevant code chunks for a query.

    Embeds the query, searches pgvector, and returns chunks ranked by similarity,
    respecting a total token budget.
    """
    query_embedding = await embed_single(query)
    embedding_str = format_embedding_for_pg(query_embedding)

    results = await db.search(
        embedding_str,
        project,
        limit=limit,
        min_score=min_score,
    )

    # Enforce token budget
    selected: list[ChunkRow] = []
    total_tokens = 0
    for chunk in results:
        if total_tokens + chunk.token_count > max_tokens:
            break
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
