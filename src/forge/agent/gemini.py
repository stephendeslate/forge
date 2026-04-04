"""Gemini cloud reasoning integration."""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

# Rate limit tracking — module-level state
_rate_limited_until: float = 0.0


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


def is_gemini_critique_available() -> bool:
    """Check if Gemini critique is available — independent of gemini.enabled.

    Requires: gemini.critique_model is set AND API key exists.
    Does NOT depend on the global gemini.enabled flag, so users can
    run local-only main loop but still use cloud critique.
    """
    from forge.config import settings

    if not settings.gemini.critique_model:
        return False
    if is_rate_limited():
        return False
    return _ensure_api_key() is not None


def is_rate_limited() -> bool:
    """Check if Gemini is currently rate-limited."""
    return time.monotonic() < _rate_limited_until


def mark_rate_limited(retry_after: float = 60.0) -> None:
    """Mark Gemini as rate-limited for the given duration (seconds)."""
    global _rate_limited_until
    _rate_limited_until = time.monotonic() + retry_after
    logger.info("Gemini rate-limited for %.0fs", retry_after)
