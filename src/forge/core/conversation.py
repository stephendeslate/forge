"""In-memory conversation history management with optional DB persistence."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from forge.log import get_logger

if TYPE_CHECKING:
    from forge.storage.database import Database

logger = get_logger(__name__)


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    model: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class Conversation:
    """Manages conversation history with a configurable max length."""

    def __init__(
        self,
        max_turns: int = 50,
        session_id: str | None = None,
        db: Database | None = None,
    ) -> None:
        self._messages: list[Message] = []
        self._max_turns = max_turns
        self.session_id = session_id
        self._db = db
        self._title_set = False

    def add(self, role: str, content: str, model: str = "") -> None:
        self._messages.append(Message(role=role, content=content, model=model))
        # Trim oldest messages if over limit (keep pairs)
        while len(self._messages) > self._max_turns * 2:
            self._messages.pop(0)

        # Fire-and-forget DB persistence
        if self._db and self.session_id:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._persist(role, content, model))
            except RuntimeError:
                logger.debug("No event loop for persistence")

    async def _persist(self, role: str, content: str, model: str) -> None:
        """Persist message to DB and auto-title on first user message."""
        assert self._db and self.session_id
        try:
            await self._db.save_message(self.session_id, role, content, model)
            # Auto-title from first user message
            if role == "user" and not self._title_set:
                title = content[:60].strip()
                if len(content) > 60:
                    title = title.rsplit(" ", 1)[0] + "…"
                await self._db.update_session_title(self.session_id, title)
                self._title_set = True
        except Exception:
            logger.debug("DB write failed", exc_info=True)

    @classmethod
    async def load_from_db(
        cls, session_id: str, db: Database, max_turns: int = 50
    ) -> Conversation:
        """Restore a conversation from PostgreSQL."""
        conv = cls(max_turns=max_turns, session_id=session_id, db=db)
        conv._title_set = True  # Don't re-title on resume
        rows = await db.load_messages(session_id)
        for row in rows:
            conv._messages.append(
                Message(
                    role=row["role"],
                    content=row["content"],
                    model=row.get("model") or "",
                    timestamp=row.get("created_at", datetime.now()),
                )
            )
        return conv

    def format_history(self) -> str:
        """Format conversation history for inclusion in prompts."""
        if not self._messages:
            return ""
        lines = []
        for msg in self._messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._messages.clear()

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def turn_count(self) -> int:
        return len([m for m in self._messages if m.role == "user"])

    def __len__(self) -> int:
        return len(self._messages)
