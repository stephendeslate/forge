"""Exemplar learning persistence."""

from __future__ import annotations

import asyncpg

from forge.storage.database import ExemplarRow


class ExemplarStore:
    """Cloud model exemplar operations: save, search, prune, outcome tracking."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save_exemplar(
        self, project: str, task_type: str, task_description: str,
        solution_approach: str, outcome_score: float, model_source: str, embedding: str,
    ) -> int:
        row = await self._pool.fetchrow(
            """
            INSERT INTO exemplars (project, task_type, task_description, solution_approach,
                                   outcome_score, model_source, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7::vector)
            RETURNING id
            """,
            project, task_type, task_description, solution_approach,
            outcome_score, model_source, embedding,
        )
        return row["id"]

    async def search_exemplars(
        self, embedding: str, project: str, *,
        task_type: str | None = None, limit: int = 3, min_score: float = 0.3,
    ) -> list[ExemplarRow]:
        if task_type:
            rows = await self._pool.fetch(
                """
                SELECT id, project, task_type, task_description, solution_approach,
                       outcome_score, model_source, created_at, used_count, last_used_at,
                       1 - (embedding <=> $1::vector) AS score
                FROM exemplars
                WHERE project = $2 AND task_type = $3
                  AND 1 - (embedding <=> $1::vector) >= $5
                ORDER BY outcome_score DESC, embedding <=> $1::vector
                LIMIT $4
                """,
                embedding, project, task_type, limit, min_score,
            )
        else:
            rows = await self._pool.fetch(
                """
                SELECT id, project, task_type, task_description, solution_approach,
                       outcome_score, model_source, created_at, used_count, last_used_at,
                       1 - (embedding <=> $1::vector) AS score
                FROM exemplars
                WHERE project = $2
                  AND 1 - (embedding <=> $1::vector) >= $4
                ORDER BY outcome_score DESC, embedding <=> $1::vector
                LIMIT $3
                """,
                embedding, project, limit, min_score,
            )
        return [ExemplarRow.from_row(r) for r in rows]

    async def update_exemplar_outcome(self, exemplar_id: int, success: bool) -> None:
        new_value = 1.0 if success else 0.0
        await self._pool.execute(
            """
            UPDATE exemplars
            SET outcome_score = 0.7 * outcome_score + 0.3 * $2
            WHERE id = $1
            """,
            exemplar_id, new_value,
        )

    async def increment_exemplar_usage(self, exemplar_id: int) -> None:
        await self._pool.execute(
            "UPDATE exemplars SET used_count = used_count + 1, last_used_at = NOW() WHERE id = $1",
            exemplar_id,
        )

    async def count_exemplars(self, project: str) -> int:
        row = await self._pool.fetchrow(
            "SELECT count(*) FROM exemplars WHERE project = $1", project,
        )
        return row["count"]

    async def list_exemplars(self, project: str, *, limit: int = 20) -> list[ExemplarRow]:
        rows = await self._pool.fetch(
            """
            SELECT id, project, task_type, task_description, solution_approach,
                   outcome_score, model_source, created_at, used_count, last_used_at
            FROM exemplars
            WHERE project = $1
            ORDER BY outcome_score DESC, created_at DESC
            LIMIT $2
            """,
            project, limit,
        )
        return [ExemplarRow.from_row(r) for r in rows]

    async def get_exemplar(self, exemplar_id: int) -> ExemplarRow | None:
        row = await self._pool.fetchrow(
            """
            SELECT id, project, task_type, task_description, solution_approach,
                   outcome_score, model_source, created_at, used_count, last_used_at
            FROM exemplars WHERE id = $1
            """,
            exemplar_id,
        )
        if not row:
            return None
        return ExemplarRow.from_row(row)

    async def delete_exemplar(self, exemplar_id: int) -> bool:
        result = await self._pool.execute("DELETE FROM exemplars WHERE id = $1", exemplar_id)
        return result == "DELETE 1"

    async def prune_exemplars(self, project: str, *, keep: int = 100) -> int:
        result = await self._pool.execute(
            """
            DELETE FROM exemplars WHERE id IN (
                SELECT id FROM exemplars
                WHERE project = $1
                ORDER BY outcome_score DESC, created_at DESC
                OFFSET $2
            )
            """,
            project, keep,
        )
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0
