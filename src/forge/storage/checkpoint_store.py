"""Conversation checkpoint persistence."""

from __future__ import annotations

from typing import Any

import asyncpg


class CheckpointStore:
    """Named save/restore points within a session."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save_checkpoint(
        self, session_id: str, name: str, agent_history: str,
        task_store: str | None, message_count: int,
    ) -> None:
        await self._pool.execute(
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
        row = await self._pool.fetchrow(
            "SELECT agent_history, task_store, message_count, created_at "
            "FROM checkpoints WHERE session_id = $1 AND name = $2",
            session_id, name,
        )
        return dict(row) if row else None

    async def list_checkpoints(self, session_id: str) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            "SELECT name, message_count, created_at "
            "FROM checkpoints WHERE session_id = $1 ORDER BY created_at DESC",
            session_id,
        )
        return [dict(r) for r in rows]

    async def delete_checkpoint(self, session_id: str, name: str) -> bool:
        result = await self._pool.execute(
            "DELETE FROM checkpoints WHERE session_id = $1 AND name = $2",
            session_id, name,
        )
        return result == "DELETE 1"
