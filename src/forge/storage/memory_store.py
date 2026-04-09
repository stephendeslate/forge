"""Cross-session memory persistence."""

from __future__ import annotations

import asyncpg

from forge.storage.database import MemoryRow


class MemoryStore:
    """Episodic memory operations: save, search, prune, merge."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save_memory(
        self, project: str, category: str, subject: str, content: str, embedding: str,
    ) -> int:
        row = await self._pool.fetchrow(
            """
            INSERT INTO memories (project, category, subject, content, embedding)
            VALUES ($1, $2, $3, $4, $5::vector)
            RETURNING id
            """,
            project, category, subject, content, embedding,
        )
        return row["id"]

    async def search_memories(
        self, embedding: str, project: str, *,
        category: str | None = None, limit: int = 10, min_score: float = 0.3,
    ) -> list[MemoryRow]:
        if category:
            rows = await self._pool.fetch(
                """
                SELECT id, project, category, subject, content, created_at, accessed_at,
                       1 - (embedding <=> $1::vector) AS score
                FROM memories
                WHERE project = $2 AND category = $3
                  AND 1 - (embedding <=> $1::vector) >= $5
                ORDER BY embedding <=> $1::vector
                LIMIT $4
                """,
                embedding, project, category, limit, min_score,
            )
        else:
            rows = await self._pool.fetch(
                """
                SELECT id, project, category, subject, content, created_at, accessed_at,
                       1 - (embedding <=> $1::vector) AS score
                FROM memories
                WHERE project = $2
                  AND 1 - (embedding <=> $1::vector) >= $4
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                embedding, project, limit, min_score,
            )

        if rows:
            ids = [r["id"] for r in rows]
            await self._pool.execute(
                "UPDATE memories SET accessed_at = NOW(), "
                "access_count = COALESCE(access_count, 0) + 1 "
                "WHERE id = ANY($1::bigint[])",
                ids,
            )

        return [MemoryRow.from_row(r) for r in rows]

    async def list_memories(
        self, project: str, *, category: str | None = None, limit: int = 50,
    ) -> list[MemoryRow]:
        if category:
            rows = await self._pool.fetch(
                "SELECT id, project, category, subject, content, created_at, accessed_at "
                "FROM memories WHERE project = $1 AND category = $2 "
                "ORDER BY created_at DESC LIMIT $3",
                project, category, limit,
            )
        else:
            rows = await self._pool.fetch(
                "SELECT id, project, category, subject, content, created_at, accessed_at "
                "FROM memories WHERE project = $1 "
                "ORDER BY created_at DESC LIMIT $2",
                project, limit,
            )
        return [MemoryRow.from_row(r) for r in rows]

    async def delete_memory(self, memory_id: int) -> bool:
        result = await self._pool.execute("DELETE FROM memories WHERE id = $1", memory_id)
        return result == "DELETE 1"

    async def count_memories(self, project: str) -> int:
        row = await self._pool.fetchrow(
            "SELECT count(*) FROM memories WHERE project = $1", project,
        )
        return row["count"]

    async def prune_memories(self, project: str, *, keep: int = 50) -> int:
        result = await self._pool.execute(
            """
            DELETE FROM memories WHERE id IN (
                SELECT id FROM memories
                WHERE project = $1
                ORDER BY accessed_at DESC
                OFFSET $2
            )
            """,
            project, keep,
        )
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    async def find_similar_pairs(
        self, project: str, threshold: float = 0.92,
    ) -> list[tuple[int, int, float]]:
        rows = await self._pool.fetch(
            """
            SELECT a.id AS id_a, b.id AS id_b,
                   1 - (a.embedding <=> b.embedding) AS similarity
            FROM memories a, memories b
            WHERE a.project = $1 AND b.project = $1
              AND a.id < b.id
              AND 1 - (a.embedding <=> b.embedding) >= $2
            ORDER BY similarity DESC
            """,
            project, threshold,
        )
        return [(r["id_a"], r["id_b"], r["similarity"]) for r in rows]

    async def get_memories_by_ids(self, ids: list[int]) -> list[MemoryRow]:
        rows = await self._pool.fetch(
            "SELECT id, project, category, subject, content, created_at, accessed_at, "
            "COALESCE(access_count, 0) AS access_count "
            "FROM memories WHERE id = ANY($1::bigint[]) ORDER BY id",
            ids,
        )
        return [MemoryRow.from_row(r) for r in rows]

    async def get_all_memories_with_embeddings(self, project: str) -> list[MemoryRow]:
        rows = await self._pool.fetch(
            "SELECT id, project, category, subject, content, created_at, accessed_at, "
            "COALESCE(access_count, 0) AS access_count "
            "FROM memories WHERE project = $1 ORDER BY id",
            project,
        )
        return [MemoryRow.from_row(r) for r in rows]

    async def merge_memory(
        self, keep_id: int, discard_id: int, new_content: str, new_embedding: str,
    ) -> None:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    UPDATE memories SET
                        content = $2,
                        embedding = $3::vector,
                        access_count = COALESCE(access_count, 0) + (
                            SELECT COALESCE(access_count, 0) FROM memories WHERE id = $4
                        )
                    WHERE id = $1
                    """,
                    keep_id, new_content, new_embedding, discard_id,
                )
                await conn.execute("DELETE FROM memories WHERE id = $1", discard_id)

    async def prune_by_ids(self, ids: list[int]) -> int:
        if not ids:
            return 0
        result = await self._pool.execute(
            "DELETE FROM memories WHERE id = ANY($1::bigint[])", ids,
        )
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0
