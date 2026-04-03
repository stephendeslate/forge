"""Gemini cloud reasoning integration."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _ensure_api_key() -> str | None:
    """Resolve API key from config or env. Returns key or None."""
    from forge.config import settings

    api_key = settings.gemini.api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return None

    # pydantic-ai GoogleModel reads GOOGLE_API_KEY from env
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = api_key

    return api_key


def get_gemini_model_string(fallback: bool = False) -> str | None:
    """Return the pydantic-ai model string for Gemini, or None if unavailable.

    Args:
        fallback: If True, return the fallback model string instead of primary.
    """
    from forge.config import settings

    if not _ensure_api_key():
        logger.warning("Gemini enabled but no API key found")
        return None

    model = settings.gemini.fallback_model if fallback else settings.gemini.model
    return f"google-gla:{model}"


def get_gemini_model_settings() -> dict | None:
    """Return model_settings dict for Gemini (timeout), or None."""
    from forge.config import settings

    timeout = settings.gemini.timeout
    if timeout:
        return {"timeout": timeout}
    return None


def is_gemini_available(deps) -> bool:
    """Check if cloud reasoning is enabled and API key is configured."""
    if not deps.cloud_reasoning_enabled:
        return False
    return _ensure_api_key() is not None
