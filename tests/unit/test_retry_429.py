"""Unit tests for 429 retryable status handling in forge.models.retry."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from forge.models.retry import _is_retryable, with_retry


class TestIsRetryable:
    """Test _is_retryable() for various exception types."""

    def test_connection_error_is_retryable(self):
        assert _is_retryable(ConnectionError("refused")) is True

    def test_os_error_is_retryable(self):
        assert _is_retryable(OSError("network down")) is True

    def test_generic_exception_is_not_retryable(self):
        assert _is_retryable(ValueError("bad value")) is False

    def test_runtime_error_is_not_retryable(self):
        assert _is_retryable(RuntimeError("boom")) is False

    def test_429_rate_limit_string_match(self):
        """Pydantic-ai wraps HTTP 429 as generic exceptions with message."""
        exc = Exception("HTTP 429: rate limit exceeded")
        assert _is_retryable(exc) is True

    def test_429_without_rate_keyword_not_retryable(self):
        """429 in string without 'rate' keyword should not match."""
        exc = Exception("error code 429")
        assert _is_retryable(exc) is False

    def test_503_service_unavailable_string_match(self):
        exc = Exception("HTTP 503: Service Unavailable")
        assert _is_retryable(exc) is True

    def test_503_overloaded_string_match(self):
        exc = Exception("503 model overloaded")
        assert _is_retryable(exc) is True

    def test_connection_refused_string_match(self):
        exc = Exception("Connection refused by remote host")
        assert _is_retryable(exc) is True

    def test_connection_reset_string_match(self):
        exc = Exception("Connection reset by peer")
        assert _is_retryable(exc) is True

    def test_httpx_429_status_error(self):
        """Test httpx.HTTPStatusError with 429 status code."""
        try:
            import httpx
        except ImportError:
            pytest.skip("httpx not installed")

        response = MagicMock()
        response.status_code = 429
        exc = httpx.HTTPStatusError("rate limited", request=MagicMock(), response=response)
        assert _is_retryable(exc) is True

    def test_httpx_503_status_error(self):
        """Test httpx.HTTPStatusError with 503 status code."""
        try:
            import httpx
        except ImportError:
            pytest.skip("httpx not installed")

        response = MagicMock()
        response.status_code = 503
        exc = httpx.HTTPStatusError("unavailable", request=MagicMock(), response=response)
        assert _is_retryable(exc) is True

    def test_httpx_read_timeout_not_retryable(self):
        """ReadTimeout should NOT be retried (avoids 3x300s stalls)."""
        try:
            import httpx
        except ImportError:
            pytest.skip("httpx not installed")

        exc = httpx.ReadTimeout("read timed out")
        assert _is_retryable(exc) is False

    def test_httpx_connect_error_is_retryable(self):
        try:
            import httpx
        except ImportError:
            pytest.skip("httpx not installed")

        exc = httpx.ConnectError("connection failed")
        assert _is_retryable(exc) is True


class TestWithRetry:
    """Test with_retry() exponential backoff logic."""

    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await with_retry(factory, max_retries=3)
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self):
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("refused")
            return "recovered"

        async def fast_sleep(delay):
            pass

        with patch("forge.models.retry.asyncio.sleep", side_effect=fast_sleep):
            result = await with_retry(factory, max_retries=3, backoff_base=0.01)

        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_on_non_retryable(self):
        async def factory():
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            await with_retry(factory, max_retries=3)

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("always fails")

        async def fast_sleep(delay):
            pass

        with patch("forge.models.retry.asyncio.sleep", side_effect=fast_sleep):
            with pytest.raises(ConnectionError, match="always fails"):
                await with_retry(factory, max_retries=2, backoff_base=0.01)

        assert call_count == 3  # initial + 2 retries
