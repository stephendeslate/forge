"""Anthropic cloud model integration — Opus via proxy."""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

# Rate limit tracking — module-level state
_rate_limited_until: float = 0.0


def _read_claude_oauth_token() -> str | None:
    """Read the OAuth access token from Claude Code's credentials file."""
    import json
    from pathlib import Path

    creds_path = Path.home() / ".claude" / ".credentials.json"
    try:
        data = json.loads(creds_path.read_text())
        token = data.get("claudeAiOauth", {}).get("accessToken", "")
        if token:
            return token
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return None


def _ensure_anthropic_env() -> None:
    """Set ANTHROPIC_API_KEY env var from config if not already set.

    Tries in order: existing env var → config api_key → Claude OAuth token → dummy.
    When using the Claude Code proxy, the OAuth access token is required.
    """
    from forge.config import settings

    if "ANTHROPIC_API_KEY" in os.environ and os.environ["ANTHROPIC_API_KEY"]:
        return

    api_key = settings.anthropic.api_key
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
        return

    # Try Claude Code's OAuth token (needed for the proxy)
    oauth_token = _read_claude_oauth_token()
    if oauth_token:
        os.environ["ANTHROPIC_API_KEY"] = oauth_token
        logger.debug("Using Claude OAuth token for Anthropic API key")
        return

    # Last resort — will likely fail auth but lets SDK initialize
    os.environ["ANTHROPIC_API_KEY"] = "proxy-auth"


def get_anthropic_model_string() -> str | None:
    """Return the pydantic-ai model string for Anthropic, or None if disabled."""
    from forge.config import settings

    if not settings.anthropic.enabled:
        return None

    _ensure_anthropic_env()
    return f"anthropic:{settings.anthropic.model}"


def get_anthropic_model_settings() -> dict:
    """Return model_settings dict for Anthropic (timeout + max_tokens)."""
    from forge.config import settings

    return {
        "timeout": settings.anthropic.timeout,
        "max_tokens": settings.anthropic.max_tokens,
    }


def is_anthropic_available() -> bool:
    """Check if Anthropic is enabled and not rate-limited."""
    from forge.config import settings

    if not settings.anthropic.enabled:
        return False
    if is_rate_limited():
        return False
    return True


def is_rate_limited() -> bool:
    """Check if Anthropic is currently rate-limited."""
    return time.monotonic() < _rate_limited_until


def mark_rate_limited(retry_after: float = 60.0) -> None:
    """Mark Anthropic as rate-limited for the given duration (seconds)."""
    global _rate_limited_until
    _rate_limited_until = time.monotonic() + retry_after
    logger.info("Anthropic rate-limited for %.0fs", retry_after)


async def anthropic_health_check() -> bool:
    """Quick check that the Anthropic proxy is reachable.

    Uses httpx to GET the base URL with a short timeout.
    Returns True if reachable, False otherwise.
    """
    import httpx

    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    # Strip trailing slash
    base_url = base_url.rstrip("/")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Use /v1/models endpoint — lightweight, doesn't consume tokens
            resp = await client.get(
                f"{base_url}/v1/models",
                headers={
                    "x-api-key": os.environ.get("ANTHROPIC_API_KEY", "proxy-auth"),
                    "anthropic-version": "2023-06-01",
                },
            )
            # Any response (even 401/403) means proxy is reachable
            return resp.status_code < 500
    except Exception:
        logger.debug("Anthropic health check failed", exc_info=True)
        return False


def _is_anthropic_error(e: Exception) -> bool:
    """Check if an exception is an Anthropic cloud error that should trigger fallback."""
    # Check pydantic-ai wrapped HTTP errors for Anthropic models
    try:
        from pydantic_ai.exceptions import ModelHTTPError

        if isinstance(e, ModelHTTPError):
            if "claude" in (e.model_name or "").lower() or "anthropic" in (e.model_name or "").lower():
                return True
    except ImportError:
        pass

    # Check raw anthropic SDK errors
    try:
        import anthropic

        if isinstance(e, (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
        )):
            return True
        if isinstance(e, anthropic.APIStatusError):
            # Rate limit, overloaded, auth errors, server errors
            if e.status_code in (401, 403, 429, 500, 529):
                return True
    except ImportError:
        pass

    # Check string-based detection for wrapped errors
    err_str = str(e).lower()
    if "anthropic" in err_str and any(
        x in err_str for x in ("connection", "timeout", "rate limit", "overloaded", "529")
    ):
        return True

    return False
