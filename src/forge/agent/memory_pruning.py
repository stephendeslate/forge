"""Smart memory pruning — composite scoring with deduplication.

Replaces simple LRU eviction with a scoring function that considers
recency, frequency, category importance, and uniqueness. Before pruning,
detects and merges semantically duplicate memories.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

from forge.log import get_logger

if TYPE_CHECKING:
    from forge.storage.database import Database

logger = get_logger(__name__)

CATEGORY_WEIGHTS = {
    "feedback": 1.0,
    "project": 0.75,
    "user": 0.5,
    "reference": 0.25,
}

# Score component weights
SCORE_WEIGHTS = {
    "recency": 0.3,
    "frequency": 0.2,
    "category": 0.3,
    "uniqueness": 0.2,
}

RECENCY_HALF_LIFE_DAYS = 14


async def smart_prune(
    db: Database,
    project: str,
    *,
    keep: int = 50,
    category_weights: dict[str, float] | None = None,
    similarity_threshold: float = 0.92,
    max_merges: int = 5,
) -> tuple[int, int]:
    """Deduplicate then score-prune. Returns (merged_count, pruned_count)."""
    cat_weights = category_weights or CATEGORY_WEIGHTS

    # Phase 1: Deduplication
    merged_count = 0
    try:
        pairs = await db.find_similar_pairs(project, similarity_threshold)
        merged_ids: set[int] = set()

        for id_a, id_b, similarity in pairs:
            if merged_count >= max_merges:
                break
            if id_a in merged_ids or id_b in merged_ids:
                continue

            # Fetch both memories to decide which to keep
            memories = await db.get_memories_by_ids([id_a, id_b])
            if len(memories) < 2:
                continue

            mem_a = memories[0]
            mem_b = memories[1]

            # Keep the one with higher access_count, tie-break by newer created_at
            ac_a = getattr(mem_a, "access_count", 0) or 0
            ac_b = getattr(mem_b, "access_count", 0) or 0

            if ac_a > ac_b or (ac_a == ac_b and (mem_a.created_at or 0) >= (mem_b.created_at or 0)):
                keep_mem, discard_mem = mem_a, mem_b
            else:
                keep_mem, discard_mem = mem_b, mem_a

            await _merge_pair(db, keep_mem, discard_mem)
            merged_ids.add(discard_mem.id)
            merged_count += 1
    except Exception:
        logger.debug("Dedup phase failed — continuing with prune", exc_info=True)

    # Phase 2: Score-based pruning
    pruned_count = 0
    try:
        all_memories = await db.get_all_memories_with_embeddings(project)
        if len(all_memories) <= keep:
            return merged_count, 0

        now = time.time()
        max_access = max((getattr(m, "access_count", 0) or 0) for m in all_memories) or 1

        # Compute pairwise max-similarity for uniqueness
        uniqueness_map: dict[int, float] = {}
        embeddings = {m.id: m.embedding for m in all_memories if hasattr(m, "embedding") and m.embedding}

        for m in all_memories:
            if m.id not in embeddings:
                uniqueness_map[m.id] = 1.0
                continue
            max_sim = 0.0
            for other in all_memories:
                if other.id == m.id or other.id not in embeddings:
                    continue
                # Use the DB-provided similarity if available, otherwise default
                max_sim = max(max_sim, 0.5)  # fallback
            uniqueness_map[m.id] = 1.0 - max_sim

        # If we have embeddings, compute actual pairwise similarity
        if len(embeddings) >= 2:
            try:
                sim_pairs = await db.find_similar_pairs(project, 0.0)
                sim_map: dict[int, float] = {}
                for id_a, id_b, sim in sim_pairs:
                    sim_map.setdefault(id_a, 0.0)
                    sim_map.setdefault(id_b, 0.0)
                    if sim > sim_map[id_a]:
                        sim_map[id_a] = sim
                    if sim > sim_map[id_b]:
                        sim_map[id_b] = sim
                for m in all_memories:
                    if m.id in sim_map:
                        uniqueness_map[m.id] = 1.0 - sim_map[m.id]
            except Exception:
                logger.debug("Pairwise similarity failed — using defaults", exc_info=True)

        # Score each memory
        scored: list[tuple[int, float]] = []
        for m in all_memories:
            score = _composite_score(
                m, now, max_access, cat_weights, uniqueness_map.get(m.id, 0.5),
            )
            scored.append((m.id, score))

        # Sort ascending — lowest scores get pruned
        scored.sort(key=lambda x: x[1])

        to_remove = len(all_memories) - keep
        if to_remove > 0:
            prune_ids = [mid for mid, _ in scored[:to_remove]]
            pruned_count = await db.prune_by_ids(prune_ids)
            logger.debug("Smart pruned %d memories for project %s", pruned_count, project)
    except Exception:
        logger.debug("Score-prune phase failed", exc_info=True)

    return merged_count, pruned_count


async def _merge_pair(db: Database, keep_mem, discard_mem) -> None:
    """Merge discard content into keeper, re-embed, delete discard."""
    from forge.models.embeddings import embed_single, format_embedding_for_pg

    new_content = f"{keep_mem.content}\n\n---\n(merged) {discard_mem.content}"
    new_embedding = await embed_single(f"{keep_mem.subject}: {new_content}")
    embedding_str = format_embedding_for_pg(new_embedding)

    await db.merge_memory(keep_mem.id, discard_mem.id, new_content, embedding_str)
    logger.debug("Merged memory #%d into #%d", discard_mem.id, keep_mem.id)


def _composite_score(
    memory,
    now: float,
    max_access_count: int,
    category_weights: dict[str, float],
    uniqueness: float,
) -> float:
    """Compute composite score. Higher = more worth keeping."""
    # Recency: exponential decay from accessed_at
    accessed_at = memory.accessed_at
    if accessed_at:
        if hasattr(accessed_at, "timestamp"):
            age_seconds = now - accessed_at.timestamp()
        else:
            age_seconds = 0.0
    else:
        # Never accessed — use created_at
        created_at = memory.created_at
        if created_at and hasattr(created_at, "timestamp"):
            age_seconds = now - created_at.timestamp()
        else:
            age_seconds = 30 * 86400  # assume 30 days old

    age_days = age_seconds / 86400
    recency = math.exp(-0.693 * age_days / RECENCY_HALF_LIFE_DAYS)  # ln(2) ~ 0.693

    # Frequency
    access_count = getattr(memory, "access_count", 0) or 0
    frequency = max(access_count / max_access_count, 0.1)

    # Category importance
    category = getattr(memory, "category", "reference")
    cat_score = category_weights.get(category, 0.25)

    # Weighted sum
    w = SCORE_WEIGHTS
    return (
        w["recency"] * recency
        + w["frequency"] * frequency
        + w["category"] * cat_score
        + w["uniqueness"] * uniqueness
    )
