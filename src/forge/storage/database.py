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
    access_count: int = 0

    @classmethod
    def from_row(cls, r: asyncpg.Record) -> MemoryRow:
        keys = r.keys()
        return cls(
            id=r["id"],
            project=r["project"],
            category=r["category"],
            subject=r["subject"],
            content=r["content"],
            created_at=r["created_at"],
            accessed_at=r["accessed_at"],
            score=r["score"] if "score" in keys else 0.0,
            access_count=r["access_count"] if "access_count" in keys else 0,
        )


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

    @classmethod
    def from_row(cls, r: asyncpg.Record) -> ChunkRow:
        return cls(
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
            score=r["score"] if "score" in r.keys() else 0.0,
        )


@dataclass
class ExemplarRow:
    id: int
    project: str
    task_type: str
    task_description: str
    solution_approach: str
    outcome_score: float
    model_source: str
    created_at: Any = None
    used_count: int = 0
    last_used_at: Any = None
    score: float = 0.0  # cosine similarity from search

    @classmethod
    def from_row(cls, r: asyncpg.Record) -> ExemplarRow:
        keys = r.keys()
        return cls(
            id=r["id"],
            project=r["project"],
            task_type=r["task_type"],
            task_description=r["task_description"],
            solution_approach=r["solution_approach"],
            outcome_score=r["outcome_score"],
            model_source=r["model_source"],
            created_at=r["created_at"],
            used_count=r["used_count"],
            last_used_at=r["last_used_at"],
            score=r["score"] if "score" in keys else 0.0,
        )


class Database:
    """Facade over focused store classes. Maintains backward compatibility."""

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

        # Stores — lazily initialized from pool
        self._chunks_store: ChunkStore | None = None
        self._sessions_store: SessionStore | None = None
        self._memories_store: MemoryStore | None = None
        self._exemplars_store: ExemplarStore | None = None
        self._checkpoints_store: CheckpointStore | None = None
        self._tasks_store: TaskStoreDB | None = None

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
            except (OSError, asyncpg.PostgresError):
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

    def _get_store(self, attr: str, cls: type):
        """Lazily initialize a store from the pool."""
        store = getattr(self, attr, None)
        if store is None:
            store = cls(self.pool)
            setattr(self, attr, store)
        return store

    @property
    def _chunks(self) -> ChunkStore:
        return self._get_store("_chunks_store", ChunkStore)

    @property
    def _sessions(self) -> SessionStore:
        return self._get_store("_sessions_store", SessionStore)

    @property
    def _memories(self) -> MemoryStore:
        return self._get_store("_memories_store", MemoryStore)

    @property
    def _exemplars(self) -> ExemplarStore:
        return self._get_store("_exemplars_store", ExemplarStore)

    @property
    def _checkpoints(self) -> CheckpointStore:
        return self._get_store("_checkpoints_store", CheckpointStore)

    @property
    def _tasks(self) -> TaskStoreDB:
        return self._get_store("_tasks_store", TaskStoreDB)

    # --- Chunk / RAG ---

    async def get_file_hash(self, project: str, file_path: str) -> str | None:
        return await self._chunks.get_file_hash(project, file_path)

    async def delete_file_chunks(self, project: str, file_path: str) -> int:
        return await self._chunks.delete_file_chunks(project, file_path)

    async def insert_chunks(self, chunks: list[dict[str, Any]]) -> int:
        return await self._chunks.insert_chunks(chunks)

    async def search(
        self, embedding: str, project: str, *, limit: int = 10, min_score: float = 0.3,
    ) -> list[ChunkRow]:
        return await self._chunks.search(embedding, project, limit=limit, min_score=min_score)

    async def text_search(self, query: str, project: str, *, limit: int = 20) -> list[ChunkRow]:
        return await self._chunks.text_search(query, project, limit=limit)

    async def get_project_stats(self, project: str) -> dict[str, Any]:
        return await self._chunks.get_project_stats(project)

    # --- Session / Conversation ---

    async def create_session(self, session_id: str, mode: str = "chat", project: str | None = None) -> None:
        await self._sessions.create_session(session_id, mode, project)

    async def update_session_title(self, session_id: str, title: str) -> None:
        await self._sessions.update_session_title(session_id, title)

    async def save_message(self, session_id: str, role: str, content: str, model: str = "") -> None:
        await self._sessions.save_message(session_id, role, content, model)

    async def load_messages(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return await self._sessions.load_messages(session_id, limit)

    async def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        return await self._sessions.list_sessions(limit)

    async def delete_session(self, session_id: str) -> None:
        await self._sessions.delete_session(session_id)

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        return await self._sessions.get_session(session_id)

    async def get_latest_session_id(self) -> str | None:
        return await self._sessions.get_latest_session_id()

    async def get_session_count(self) -> int:
        return await self._sessions.get_session_count()

    async def delete_agent_history(self, session_id: str) -> None:
        await self._sessions.delete_agent_history(session_id)

    async def load_agent_history(self, session_id: str) -> str | None:
        return await self._sessions.load_agent_history(session_id)

    # --- Memory ---

    async def save_memory(self, project: str, category: str, subject: str, content: str, embedding: str) -> int:
        return await self._memories.save_memory(project, category, subject, content, embedding)

    async def search_memories(
        self, embedding: str, project: str, *, category: str | None = None,
        limit: int = 10, min_score: float = 0.3,
    ) -> list[MemoryRow]:
        return await self._memories.search_memories(
            embedding, project, category=category, limit=limit, min_score=min_score,
        )

    async def list_memories(self, project: str, *, category: str | None = None, limit: int = 50) -> list[MemoryRow]:
        return await self._memories.list_memories(project, category=category, limit=limit)

    async def delete_memory(self, memory_id: int) -> bool:
        return await self._memories.delete_memory(memory_id)

    async def count_memories(self, project: str) -> int:
        return await self._memories.count_memories(project)

    async def prune_memories(self, project: str, *, keep: int = 50) -> int:
        return await self._memories.prune_memories(project, keep=keep)

    async def find_similar_pairs(self, project: str, threshold: float = 0.92) -> list[tuple[int, int, float]]:
        return await self._memories.find_similar_pairs(project, threshold)

    async def get_memories_by_ids(self, ids: list[int]) -> list[MemoryRow]:
        return await self._memories.get_memories_by_ids(ids)

    async def get_all_memories_with_embeddings(self, project: str) -> list[MemoryRow]:
        return await self._memories.get_all_memories_with_embeddings(project)

    async def merge_memory(self, keep_id: int, discard_id: int, new_content: str, new_embedding: str) -> None:
        await self._memories.merge_memory(keep_id, discard_id, new_content, new_embedding)

    async def prune_by_ids(self, ids: list[int]) -> int:
        return await self._memories.prune_by_ids(ids)

    # --- Exemplar ---

    async def save_exemplar(
        self, project: str, task_type: str, task_description: str,
        solution_approach: str, outcome_score: float, model_source: str, embedding: str,
    ) -> int:
        return await self._exemplars.save_exemplar(
            project, task_type, task_description, solution_approach,
            outcome_score, model_source, embedding,
        )

    async def search_exemplars(
        self, embedding: str, project: str, *, task_type: str | None = None,
        limit: int = 3, min_score: float = 0.3,
    ) -> list[ExemplarRow]:
        return await self._exemplars.search_exemplars(
            embedding, project, task_type=task_type, limit=limit, min_score=min_score,
        )

    async def update_exemplar_outcome(self, exemplar_id: int, success: bool) -> None:
        await self._exemplars.update_exemplar_outcome(exemplar_id, success)

    async def increment_exemplar_usage(self, exemplar_id: int) -> None:
        await self._exemplars.increment_exemplar_usage(exemplar_id)

    async def count_exemplars(self, project: str) -> int:
        return await self._exemplars.count_exemplars(project)

    async def list_exemplars(self, project: str, *, limit: int = 20) -> list[ExemplarRow]:
        return await self._exemplars.list_exemplars(project, limit=limit)

    async def get_exemplar(self, exemplar_id: int) -> ExemplarRow | None:
        return await self._exemplars.get_exemplar(exemplar_id)

    async def delete_exemplar(self, exemplar_id: int) -> bool:
        return await self._exemplars.delete_exemplar(exemplar_id)

    async def prune_exemplars(self, project: str, *, keep: int = 100) -> int:
        return await self._exemplars.prune_exemplars(project, keep=keep)

    # --- Task store ---

    async def save_task_store(self, session_id: str, task_json: str) -> None:
        await self._tasks.save_task_store(session_id, task_json)

    async def load_task_store(self, session_id: str) -> str | None:
        return await self._tasks.load_task_store(session_id)

    # --- Checkpoints ---

    async def save_checkpoint(
        self, session_id: str, name: str, agent_history: str,
        task_store: str | None, message_count: int,
    ) -> None:
        await self._checkpoints.save_checkpoint(session_id, name, agent_history, task_store, message_count)

    async def load_checkpoint(self, session_id: str, name: str) -> dict[str, Any] | None:
        return await self._checkpoints.load_checkpoint(session_id, name)

    async def list_checkpoints(self, session_id: str) -> list[dict[str, Any]]:
        return await self._checkpoints.list_checkpoints(session_id)

    async def delete_checkpoint(self, session_id: str, name: str) -> bool:
        return await self._checkpoints.delete_checkpoint(session_id, name)


# Import stores at bottom to avoid circular imports
from forge.storage.checkpoint_store import CheckpointStore  # noqa: E402
from forge.storage.chunk_store import ChunkStore  # noqa: E402
from forge.storage.exemplar_store import ExemplarStore  # noqa: E402
from forge.storage.memory_store import MemoryStore  # noqa: E402
from forge.storage.session_store import SessionStore  # noqa: E402
from forge.storage.task_store_db import TaskStoreDB  # noqa: E402
