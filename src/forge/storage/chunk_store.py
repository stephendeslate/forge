"""Chunk storage for RAG indexing and retrieval."""

from __future__ import annotations

from typing import Any

import asyncpg

from forge.storage.database import ChunkRow


class ChunkStore:
    """RAG chunk operations: insert, search, delete, stats."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_file_hash(self, project: str, file_path: str) -> str | None:
        row = await self._pool.fetchrow(
            "SELECT DISTINCT file_hash FROM chunks WHERE project = $1 AND file_path = $2",
            project, file_path,
        )
        return row["file_hash"] if row else None

    async def delete_file_chunks(self, project: str, file_path: str) -> int:
        result = await self._pool.execute(
            "DELETE FROM chunks WHERE project = $1 AND file_path = $2",
            project, file_path,
        )
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    async def insert_chunks(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0
        records = [
            (
                c["project"], c["file_path"], c["chunk_type"], c.get("name"),
                c["content"], c["start_line"], c["end_line"], c["token_count"],
                c["embedding"], c["file_hash"],
            )
            for c in chunks
        ]
        await self._pool.executemany(
            """
            INSERT INTO chunks (project, file_path, chunk_type, name, content,
                                start_line, end_line, token_count, embedding, file_hash)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector, $10)
            """,
            records,
        )
        return len(records)

    async def search(
        self, embedding: str, project: str, *, limit: int = 10, min_score: float = 0.3,
    ) -> list[ChunkRow]:
        rows = await self._pool.fetch(
            """
            SELECT id, project, file_path, chunk_type, name, content,
                   start_line, end_line, token_count, file_hash,
                   1 - (embedding <=> $1::vector) AS score
            FROM chunks
            WHERE project = $2
              AND 1 - (embedding <=> $1::vector) >= $4
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """,
            embedding, project, limit, min_score,
        )
        return [ChunkRow.from_row(r) for r in rows]

    async def text_search(
        self, query: str, project: str, *, limit: int = 20,
    ) -> list[ChunkRow]:
        rows = await self._pool.fetch(
            """
            SELECT id, project, file_path, chunk_type, name, content,
                   start_line, end_line, token_count, file_hash,
                   ts_rank(tsv, websearch_to_tsquery('english', $1)) AS score
            FROM chunks
            WHERE project = $2
              AND tsv @@ websearch_to_tsquery('english', $1)
            ORDER BY score DESC
            LIMIT $3
            """,
            query, project, limit,
        )
        return [ChunkRow.from_row(r) for r in rows]

    async def get_project_stats(self, project: str) -> dict[str, Any]:
        row = await self._pool.fetchrow(
            """
            SELECT count(*) AS chunk_count,
                   count(DISTINCT file_path) AS file_count,
                   max(indexed_at) AS last_indexed
            FROM chunks WHERE project = $1
            """,
            project,
        )
        return dict(row) if row else {"chunk_count": 0, "file_count": 0, "last_indexed": None}
