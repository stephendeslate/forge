"""High-level memory operations — embed + DB for cross-session recall."""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge.log import get_logger
from forge.models.embeddings import embed_single, format_embedding_for_pg

if TYPE_CHECKING:
    from forge.storage.database import Database, MemoryRow

logger = get_logger(__name__)

MAX_MEMORIES = 50  # Auto-prune threshold


async def save_memory_to_db(
    db: Database,
    project: str,
    category: str,
    subject: str,
    content: str,
) -> int:
    """Embed content, insert memory, and auto-prune if over threshold."""
    embedding = await embed_single(f"{subject}: {content}")
    embedding_str = format_embedding_for_pg(embedding)

    memory_id = await db.save_memory(project, category, subject, content, embedding_str)

    # Auto-prune
    count = await db.count_memories(project)
    if count > MAX_MEMORIES:
        pruned = await db.prune_memories(project, keep=MAX_MEMORIES)
        if pruned > 0:
            logger.debug("Pruned %d old memories for project %s", pruned, project)

    return memory_id


async def recall_from_db(
    db: Database,
    project: str,
    query: str,
    *,
    category: str | None = None,
    limit: int = 5,
) -> list[MemoryRow]:
    """Embed query and search memories by similarity."""
    embedding = await embed_single(query)
    embedding_str = format_embedding_for_pg(embedding)

    return await db.search_memories(
        embedding_str, project, category=category, limit=limit,
    )


async def get_startup_memories(
    db: Database,
    project: str,
    *,
    limit: int = 10,
) -> list[MemoryRow]:
    """Get most recent memories for system prompt injection at startup."""
    return await db.list_memories(project, limit=limit)
