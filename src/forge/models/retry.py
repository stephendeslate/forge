"""Retry with exponential backoff for transient API errors."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from forge.log import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Exceptions that indicate transient connection issues worth retrying
_RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    OSError,
)

# HTTP status codes worth retrying
_RETRYABLE_STATUS_CODES = {429, 503}


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is retryable."""
    # Check direct type match
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True

    # Check httpx-specific errors (without importing at module level)
    try:
        import httpx

        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
            return True
        # Do NOT retry ReadTimeout — avoids 3x300s stalls
        if isinstance(exc, httpx.ReadTimeout):
            return False
        # Check for HTTP 503 wrapped in exceptions
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in _RETRYABLE_STATUS_CODES:
            return True
    except ImportError:
        pass

    # Check pydantic-ai wrapped errors
    exc_str = str(exc)
    if "429" in exc_str and "rate" in exc_str.lower():
        return True
    if "503" in exc_str and ("service unavailable" in exc_str.lower() or "overloaded" in exc_str.lower()):
        return True
    if "connection" in exc_str.lower() and ("refused" in exc_str.lower() or "reset" in exc_str.lower()):
        return True

    return False


async def with_retry(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    backoff_base: float = 1.0,
) -> T:
    """Execute an async callable with exponential backoff on transient errors.

    Args:
        coro_factory: A callable that returns a new awaitable each call.
        max_retries: Maximum number of retry attempts.
        backoff_base: Base delay in seconds (doubles each attempt).

    Returns:
        The result of the successful call.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            last_exc = exc

            if attempt >= max_retries or not _is_retryable(exc):
                raise

            delay = backoff_base * (2 ** attempt)
            logger.warning(
                "Transient error (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, max_retries, delay, exc,
            )
            await asyncio.sleep(delay)

    # Should never reach here, but satisfy type checker
    raise last_exc  # type: ignore[misc]
