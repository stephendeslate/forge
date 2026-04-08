"""High-level memory operations — embed + DB for cross-session recall."""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge.config import settings
from forge.log import get_logger
from forge.models.embeddings import embed_single, format_embedding_for_pg

if TYPE_CHECKING:
    from forge.storage.database import Database, MemoryRow

logger = get_logger(__name__)


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

    max_memories = settings.memory.max_memories
    memory_id = await db.save_memory(project, category, subject, content, embedding_str)

    # Auto-prune using smart scoring
    count = await db.count_memories(project)
    if count > max_memories:
        try:
            from forge.agent.memory_pruning import smart_prune

            merged, pruned = await smart_prune(
                db, project,
                keep=max_memories,
                similarity_threshold=settings.memory.similarity_threshold,
                max_merges=settings.memory.max_merges,
            )
            if merged or pruned:
                logger.debug(
                    "Smart prune for %s: merged=%d, pruned=%d", project, merged, pruned,
                )
        except Exception:
            # Fall back to simple LRU
            logger.debug("Smart prune failed, falling back to LRU", exc_info=True)
            pruned = await db.prune_memories(project, keep=max_memories)
            if pruned > 0:
                logger.debug("LRU pruned %d memories for project %s", pruned, project)

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


async def get_relevant_startup_memories(
    db: Database,
    project: str,
    query: str,
    *,
    limit: int = 10,
) -> list[MemoryRow]:
    """Get memories relevant to a context query for startup injection.

    Falls back to chronological (get_startup_memories) if query is empty
    or embedding fails.
    """
    if not query.strip():
        return await get_startup_memories(db, project, limit=limit)

    try:
        embedding = await embed_single(query)
        embedding_str = format_embedding_for_pg(embedding)
        return await db.search_memories(
            embedding_str, project, limit=limit,
        )
    except Exception:
        logger.debug("Semantic startup memory failed, falling back to chronological", exc_info=True)
        return await get_startup_memories(db, project, limit=limit)
