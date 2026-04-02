"""asyncpg + pgvector database layer for chunk storage and retrieval."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import asyncpg


@dataclass
class MemoryRow:
    id: int
    project: str
    category: str
    subject: str
    content: str
    created_at: Any = None
    accessed_at: Any = None
    score: float = 0.0


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

    def __init__(self, dsn: str | None = None) -> None:
        from forge.config import settings

        cfg = settings.db
        self._dsn = dsn or cfg.dsn
        self._pool_min = cfg.pool_min
        self._pool_max = cfg.pool_max
        self._connect_timeout = cfg.connect_timeout
        self._retry_attempts = cfg.retry_attempts
        self._retry_delay = cfg.retry_delay
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        for attempt in range(self._retry_attempts):
            try:
                self._pool = await asyncpg.create_pool(
                    self._dsn,
                    min_size=self._pool_min,
                    max_size=self._pool_max,
                    timeout=self._connect_timeout,
                )
                # Auto-run pending migrations
                try:
                    from forge.storage.migrations import run_migrations

                    applied = await run_migrations(self._pool)
                    if applied:
                        from forge.log import get_logger

                        get_logger(__name__).info(
                            "Applied %d migration(s): %s",
                            len(applied),
                            ", ".join(applied),
                        )
                except Exception:
                    from forge.log import get_logger

                    get_logger(__name__).warning(
                        "Migration check failed — continuing without migrations",
                        exc_info=True,
                    )
                return
            except (OSError, asyncpg.PostgresError) as exc:
                if attempt == self._retry_attempts - 1:
                    raise
                await asyncio.sleep(self._retry_delay * (attempt + 1))

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
        try:
            return int(result.split()[-1])  # "DELETE N"
        except (ValueError, IndexError):
            return 0

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
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO conversations (session_id, role, content, model) VALUES ($1, $2, $3, $4)",
                    session_id, role, content, model or None,
                )
                await conn.execute(
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

    # --- Memory persistence ---

    async def save_memory(
        self,
        project: str,
        category: str,
        subject: str,
        content: str,
        embedding: str,
    ) -> int:
        """Insert a memory and return its ID."""
        row = await self.pool.fetchrow(
            """
            INSERT INTO memories (project, category, subject, content, embedding)
            VALUES ($1, $2, $3, $4, $5::vector)
            RETURNING id
            """,
            project, category, subject, content, embedding,
        )
        return row["id"]

    async def search_memories(
        self,
        embedding: str,
        project: str,
        *,
        category: str | None = None,
        limit: int = 10,
        min_score: float = 0.3,
    ) -> list[MemoryRow]:
        """Search memories by vector similarity."""
        if category:
            rows = await self.pool.fetch(
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
            rows = await self.pool.fetch(
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

        # Touch accessed_at and increment access_count for returned memories
        if rows:
            ids = [r["id"] for r in rows]
            await self.pool.execute(
                "UPDATE memories SET accessed_at = NOW(), "
                "access_count = COALESCE(access_count, 0) + 1 "
                "WHERE id = ANY($1::bigint[])",
                ids,
            )

        return [
            MemoryRow(
                id=r["id"],
                project=r["project"],
                category=r["category"],
                subject=r["subject"],
                content=r["content"],
                created_at=r["created_at"],
                accessed_at=r["accessed_at"],
                score=r["score"],
            )
            for r in rows
        ]

    async def list_memories(
        self,
        project: str,
        *,
        category: str | None = None,
        limit: int = 50,
    ) -> list[MemoryRow]:
        """List memories for a project, newest first."""
        if category:
            rows = await self.pool.fetch(
                "SELECT id, project, category, subject, content, created_at, accessed_at "
                "FROM memories WHERE project = $1 AND category = $2 "
                "ORDER BY created_at DESC LIMIT $3",
                project, category, limit,
            )
        else:
            rows = await self.pool.fetch(
                "SELECT id, project, category, subject, content, created_at, accessed_at "
                "FROM memories WHERE project = $1 "
                "ORDER BY created_at DESC LIMIT $2",
                project, limit,
            )
        return [
            MemoryRow(
                id=r["id"], project=r["project"], category=r["category"],
                subject=r["subject"], content=r["content"],
                created_at=r["created_at"], accessed_at=r["accessed_at"],
            )
            for r in rows
        ]

    async def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory by ID. Returns True if deleted."""
        result = await self.pool.execute(
            "DELETE FROM memories WHERE id = $1", memory_id,
        )
        return result == "DELETE 1"

    async def count_memories(self, project: str) -> int:
        """Count memories for a project."""
        row = await self.pool.fetchrow(
            "SELECT count(*) FROM memories WHERE project = $1", project,
        )
        return row["count"]

    async def prune_memories(self, project: str, *, keep: int = 50) -> int:
        """Delete oldest memories beyond the keep limit. Returns count deleted."""
        result = await self.pool.execute(
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
        """Find memory pairs with cosine similarity >= threshold."""
        rows = await self.pool.fetch(
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
        """Fetch specific memories by ID."""
        rows = await self.pool.fetch(
            "SELECT id, project, category, subject, content, created_at, accessed_at, "
            "COALESCE(access_count, 0) AS access_count "
            "FROM memories WHERE id = ANY($1::bigint[]) ORDER BY id",
            ids,
        )
        result = []
        for r in rows:
            mr = MemoryRow(
                id=r["id"], project=r["project"], category=r["category"],
                subject=r["subject"], content=r["content"],
                created_at=r["created_at"], accessed_at=r["accessed_at"],
            )
            mr.access_count = r["access_count"]  # type: ignore[attr-defined]
            result.append(mr)
        return result

    async def get_all_memories_with_embeddings(self, project: str) -> list[MemoryRow]:
        """Fetch all memories for a project with access_count."""
        rows = await self.pool.fetch(
            "SELECT id, project, category, subject, content, created_at, accessed_at, "
            "COALESCE(access_count, 0) AS access_count "
            "FROM memories WHERE project = $1 ORDER BY id",
            project,
        )
        result = []
        for r in rows:
            mr = MemoryRow(
                id=r["id"], project=r["project"], category=r["category"],
                subject=r["subject"], content=r["content"],
                created_at=r["created_at"], accessed_at=r["accessed_at"],
            )
            mr.access_count = r["access_count"]  # type: ignore[attr-defined]
            result.append(mr)
        return result

    async def merge_memory(
        self, keep_id: int, discard_id: int, new_content: str, new_embedding: str,
    ) -> None:
        """Merge two memories: update keeper content/embedding, delete discarded."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Sum access counts
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
        """Delete memories by ID list. Returns count deleted."""
        if not ids:
            return 0
        result = await self.pool.execute(
            "DELETE FROM memories WHERE id = ANY($1::bigint[])", ids,
        )
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    # --- Task store persistence ---

    async def save_task_store(self, session_id: str, task_json: str) -> None:
        """Persist task store JSON to conversations table."""
        await self.pool.execute(
            "DELETE FROM conversations WHERE session_id = $1 AND role = 'task_store'",
            session_id,
        )
        await self.save_message(session_id, "task_store", task_json, model="")

    async def load_task_store(self, session_id: str) -> str | None:
        """Load task store JSON. Returns None if not found."""
        row = await self.pool.fetchrow(
            "SELECT content FROM conversations WHERE session_id = $1 AND role = 'task_store' "
            "ORDER BY created_at DESC LIMIT 1",
            session_id,
        )
        return row["content"] if row else None

    # --- Checkpoint persistence ---

    async def save_checkpoint(
        self,
        session_id: str,
        name: str,
        agent_history: str,
        task_store: str | None,
        message_count: int,
    ) -> None:
        """UPSERT a named checkpoint for a session."""
        await self.pool.execute(
            """
            INSERT INTO checkpoints (session_id, name, agent_history, task_store, message_count)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (session_id, name) DO UPDATE
            SET agent_history = EXCLUDED.agent_history,
                task_store = EXCLUDED.task_store,
                message_count = EXCLUDED.message_count,
                created_at = NOW()
            """,
            session_id, name, agent_history, task_store, message_count,
        )

    async def load_checkpoint(self, session_id: str, name: str) -> dict[str, Any] | None:
        """Load a named checkpoint. Returns dict with agent_history, task_store, message_count."""
        row = await self.pool.fetchrow(
            "SELECT agent_history, task_store, message_count, created_at "
            "FROM checkpoints WHERE session_id = $1 AND name = $2",
            session_id, name,
        )
        return dict(row) if row else None

    async def list_checkpoints(self, session_id: str) -> list[dict[str, Any]]:
        """List checkpoints for a session, newest first."""
        rows = await self.pool.fetch(
            "SELECT name, message_count, created_at "
            "FROM checkpoints WHERE session_id = $1 ORDER BY created_at DESC",
            session_id,
        )
        return [dict(r) for r in rows]

    async def delete_checkpoint(self, session_id: str, name: str) -> bool:
        """Delete a checkpoint by name. Returns True if deleted."""
        result = await self.pool.execute(
            "DELETE FROM checkpoints WHERE session_id = $1 AND name = $2",
            session_id, name,
        )
        return result == "DELETE 1"


