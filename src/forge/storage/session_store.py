"""Session and conversation persistence."""

from __future__ import annotations

from typing import Any

import asyncpg


class SessionStore:
    """Session lifecycle and conversation message storage."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_session(self, session_id: str, mode: str = "chat", project: str | None = None) -> None:
        await self._pool.execute(
            "INSERT INTO sessions (id, mode, project) VALUES ($1, $2, $3)",
            session_id, mode, project,
        )

    async def update_session_title(self, session_id: str, title: str) -> None:
        await self._pool.execute(
            "UPDATE sessions SET title = $1, updated_at = NOW() WHERE id = $2",
            title, session_id,
        )

    async def save_message(self, session_id: str, role: str, content: str, model: str = "") -> None:
        async with self._pool.acquire() as conn:
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
        rows = await self._pool.fetch(
            "SELECT role, content, model, created_at FROM conversations "
            "WHERE session_id = $1 ORDER BY created_at LIMIT $2",
            session_id, limit,
        )
        return [dict(r) for r in rows]

    async def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
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
        await self._pool.execute("DELETE FROM conversations WHERE session_id = $1", session_id)
        await self._pool.execute("DELETE FROM sessions WHERE id = $1", session_id)

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = await self._pool.fetchrow("SELECT * FROM sessions WHERE id = $1", session_id)
        return dict(row) if row else None

    async def get_latest_session_id(self) -> str | None:
        row = await self._pool.fetchrow("SELECT id FROM sessions ORDER BY updated_at DESC LIMIT 1")
        return row["id"] if row else None

    async def get_session_count(self) -> int:
        row = await self._pool.fetchrow("SELECT count(*) FROM sessions")
        return row["count"]

    async def delete_agent_history(self, session_id: str) -> None:
        await self._pool.execute(
            "DELETE FROM conversations WHERE session_id = $1 AND role = 'agent_history'",
            session_id,
        )

    async def load_agent_history(self, session_id: str) -> str | None:
        row = await self._pool.fetchrow(
            "SELECT content FROM conversations WHERE session_id = $1 AND role = 'agent_history' "
            "ORDER BY created_at DESC LIMIT 1",
            session_id,
        )
        return row["content"] if row else None
