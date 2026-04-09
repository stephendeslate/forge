"""Session persistence — save/load agent history and database connection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from forge.config import settings
from forge.log import get_logger

if TYPE_CHECKING:
    from forge.storage.database import Database

logger = get_logger(__name__)

_message_list_adapter = ModelMessagesTypeAdapter


async def _save_agent_session(
    db: Database,
    session_id: str,
    messages: list[ModelMessage],
) -> None:
    """Persist agent message history as JSON to the conversations table."""
    try:
        history_json = _message_list_adapter.dump_json(messages).decode()
        await db.delete_agent_history(session_id)
        await db.save_message(session_id, "agent_history", history_json, model="")
    except Exception:
        logger.debug("Failed to save session", exc_info=True)


async def _load_agent_history(
    db: Database,
    session_id: str,
) -> list[ModelMessage] | None:
    """Load agent message history from the database. Returns None if not found."""
    try:
        raw = await db.load_agent_history(session_id)
        if raw is None:
            return None
        return _message_list_adapter.validate_json(raw)
    except Exception:
        logger.debug("Failed to load history", exc_info=True)
        return None


async def _connect_db():
    """Connect to DB for persistence. Returns None on failure."""
    if not settings.persist_history:
        return None
    try:
        from forge.storage.database import Database

        db = Database()
        await db.connect()
        return db
    except Exception:
        logger.info("Database unavailable — persistence disabled", exc_info=True)
        return None
