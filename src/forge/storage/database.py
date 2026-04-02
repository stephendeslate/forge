"""asyncpg + pgvector database layer for chunk storage and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import asyncpg


@dataclass
class ChunkRow:
    id: int
    project: str
    file_path: str
    chunk_type: str
    name: str | None
    content: str
    start_line: int
    end_line: int
    token_count: int
    file_hash: str
    score: float = 0.0  # cosine similarity score from search


class Database:
    """Manages the asyncpg connection pool and chunk operations."""

    def __init__(self, dsn: str = "postgresql://stephen@/forge?host=/var/run/postgresql&port=5433") -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=2,
            max_size=10,
        )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._pool

    async def get_file_hash(self, project: str, file_path: str) -> str | None:
        """Get the stored file hash for incremental indexing. Returns None if not indexed."""
        row = await self.pool.fetchrow(
            "SELECT DISTINCT file_hash FROM chunks WHERE project = $1 AND file_path = $2",
            project,
            file_path,
        )
        return row["file_hash"] if row else None

    async def delete_file_chunks(self, project: str, file_path: str) -> int:
        """Delete all chunks for a file (before re-indexing)."""
        result = await self.pool.execute(
            "DELETE FROM chunks WHERE project = $1 AND file_path = $2",
            project,
            file_path,
        )
        return int(result.split()[-1])  # "DELETE N"

    async def insert_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Bulk insert chunks with embeddings."""
        if not chunks:
            return 0

        records = [
            (
                c["project"],
                c["file_path"],
                c["chunk_type"],
                c.get("name"),
                c["content"],
                c["start_line"],
                c["end_line"],
                c["token_count"],
                c["embedding"],
                c["file_hash"],
            )
            for c in chunks
        ]

        await self.pool.executemany(
            """
            INSERT INTO chunks (project, file_path, chunk_type, name, content,
                                start_line, end_line, token_count, embedding, file_hash)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector, $10)
            """,
            records,
        )
        return len(records)

    async def search(
        self,
        embedding: str,
        project: str,
        *,
        limit: int = 10,
        min_score: float = 0.3,
    ) -> list[ChunkRow]:
        """Search for similar chunks using cosine distance on vector."""
        rows = await self.pool.fetch(
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
            embedding,
            project,
            limit,
            min_score,
        )
        return [
            ChunkRow(
                id=r["id"],
                project=r["project"],
                file_path=r["file_path"],
                chunk_type=r["chunk_type"],
                name=r["name"],
                content=r["content"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                token_count=r["token_count"],
                file_hash=r["file_hash"],
                score=r["score"],
            )
            for r in rows
        ]

    # --- Session / conversation persistence ---

    async def create_session(self, session_id: str, mode: str = "chat", project: str | None = None) -> None:
        await self.pool.execute(
            "INSERT INTO sessions (id, mode, project) VALUES ($1, $2, $3)",
            session_id, mode, project,
        )

    async def update_session_title(self, session_id: str, title: str) -> None:
        await self.pool.execute(
            "UPDATE sessions SET title = $1, updated_at = NOW() WHERE id = $2",
            title, session_id,
        )

    async def save_message(self, session_id: str, role: str, content: str, model: str = "") -> None:
        await self.pool.execute(
            "INSERT INTO conversations (session_id, role, content, model) VALUES ($1, $2, $3, $4)",
            session_id, role, content, model or None,
        )
        await self.pool.execute(
            "UPDATE sessions SET updated_at = NOW() WHERE id = $1",
            session_id,
        )

    async def load_messages(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        rows = await self.pool.fetch(
            "SELECT role, content, model, created_at FROM conversations "
            "WHERE session_id = $1 ORDER BY created_at LIMIT $2",
            session_id, limit,
        )
        return [dict(r) for r in rows]

    async def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = await self.pool.fetch(
            """
            SELECT s.id, s.title, s.mode, s.project, s.created_at, s.updated_at,
                   count(c.id) AS message_count
            FROM sessions s
            LEFT JOIN conversations c ON c.session_id = s.id
            GROUP BY s.id
            ORDER BY s.updated_at DESC
            LIMIT $1
            """,
            limit,
        )
        return [dict(r) for r in rows]

    async def delete_session(self, session_id: str) -> None:
        await self.pool.execute("DELETE FROM conversations WHERE session_id = $1", session_id)
        await self.pool.execute("DELETE FROM sessions WHERE id = $1", session_id)

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = await self.pool.fetchrow("SELECT * FROM sessions WHERE id = $1", session_id)
        return dict(row) if row else None

    async def get_latest_session_id(self) -> str | None:
        row = await self.pool.fetchrow("SELECT id FROM sessions ORDER BY updated_at DESC LIMIT 1")
        return row["id"] if row else None

    async def get_session_count(self) -> int:
        row = await self.pool.fetchrow("SELECT count(*) FROM sessions")
        return row["count"]

    async def delete_agent_history(self, session_id: str) -> None:
        """Delete the agent_history row for a session (before re-saving)."""
        await self.pool.execute(
            "DELETE FROM conversations WHERE session_id = $1 AND role = 'agent_history'",
            session_id,
        )

    async def load_agent_history(self, session_id: str) -> str | None:
        """Load the agent_history JSON blob for a session. Returns None if not found."""
        row = await self.pool.fetchrow(
            "SELECT content FROM conversations WHERE session_id = $1 AND role = 'agent_history' "
            "ORDER BY created_at DESC LIMIT 1",
            session_id,
        )
        return row["content"] if row else None

    async def get_project_stats(self, project: str) -> dict[str, Any]:
        """Get indexing stats for a project."""
        row = await self.pool.fetchrow(
            """
            SELECT count(*) AS chunk_count,
                   count(DISTINCT file_path) AS file_count,
                   max(indexed_at) AS last_indexed
            FROM chunks WHERE project = $1
            """,
            project,
        )
        return dict(row) if row else {"chunk_count": 0, "file_count": 0, "last_indexed": None}


