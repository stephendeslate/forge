"""Task store persistence via conversations table."""

from __future__ import annotations

import asyncpg


class TaskStoreDB:
    """Persists in-memory task store JSON to the database."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save_task_store(self, session_id: str, task_json: str) -> None:
        await self._pool.execute(
            "DELETE FROM conversations WHERE session_id = $1 AND role = 'task_store'",
            session_id,
        )
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO conversations (session_id, role, content, model) VALUES ($1, $2, $3, $4)",
                    session_id, "task_store", task_json, None,
                )
                await conn.execute(
                    "UPDATE sessions SET updated_at = NOW() WHERE id = $1",
                    session_id,
                )

    async def load_task_store(self, session_id: str) -> str | None:
        row = await self._pool.fetchrow(
            "SELECT content FROM conversations WHERE session_id = $1 AND role = 'task_store' "
            "ORDER BY created_at DESC LIMIT 1",
            session_id,
        )
        return row["content"] if row else None
