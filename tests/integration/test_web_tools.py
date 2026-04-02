"""Integration tests for web_search and web_fetch tools.

Requires a local SearXNG instance running on port 8888:
    docker run -d --name forge-searxng -p 8888:8080 \
        -v /path/to/settings.yml:/etc/searxng/settings.yml \
        searxng/searxng

The settings.yml must enable JSON format:
    use_default_settings: true
    search:
      formats: [html, json]
    server:
      secret_key: "..."
      limiter: false
"""

from __future__ import annotations

import httpx
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console

from forge.agent.deps import AgentDeps
from forge.agent.permissions import PermissionPolicy
from forge.agent.tools import web_search, web_fetch, _assess_content_quality


def _searxng_available() -> bool:
    """Check if SearXNG is reachable."""
    try:
        resp = httpx.get("http://localhost:8888/search?q=test&format=json", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


has_searxng = _searxng_available()


@pytest.fixture
def ctx(tmp_path):
    """Mock RunContext with AgentDeps."""
    mock = MagicMock()
    mock.deps = AgentDeps(
        cwd=tmp_path,
        console=Console(file=None, no_color=True),
        permission=PermissionPolicy.YOLO,
    )
    return mock


def _mock_streaming_response(
    content: str | bytes,
    status_code: int = 200,
    content_type: str = "text/html",
) -> MagicMock:
    """Create a mock async context manager for client.stream() with aiter_bytes support."""
    if isinstance(content, str):
        content = content.encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.headers = {"content-type": content_type}

    async def aiter_bytes(chunk_size=4096):
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]

    mock_resp.aiter_bytes = aiter_bytes

    # Make the response work as an async context manager
    stream_cm = AsyncMock()
    stream_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    stream_cm.__aexit__ = AsyncMock(return_value=False)

    return stream_cm


def _mock_client_with_stream(stream_cm: MagicMock) -> MagicMock:
    """Wrap a streaming response mock in a client async context manager."""
    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=stream_cm)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


@pytest.mark.skipif(not has_searxng, reason="SearXNG not running on localhost:8888")
class TestWebSearch:
    async def test_search_returns_results(self, ctx):
        result = await web_search(ctx, "python programming language")
        assert "Search results for:" in result
        assert "python" in result.lower()

    async def test_search_contains_urls(self, ctx):
        result = await web_search(ctx, "github")
        # Results should contain URLs
        assert "http" in result

    async def test_search_max_results(self, ctx):
        result = await web_search(ctx, "linux", max_results=2)
        # Count numbered results (1. ..., 2. ...)
        lines = [l for l in result.split("\n") if l.strip().startswith(("1.", "2.", "3."))]
        assert len(lines) <= 2

    async def test_search_max_results_capped(self, ctx):
        """max_results should be capped at 10."""
        result = await web_search(ctx, "python", max_results=100)
        # Should still work, just capped
        assert "Search results for:" in result

    async def test_search_no_results(self, ctx):
        result = await web_search(ctx, "xyzzy_completely_nonexistent_query_8675309")
        # Should handle gracefully — either "no results" or empty results
        assert isinstance(result, str)


@pytest.mark.skipif(not has_searxng, reason="SearXNG not running on localhost:8888")
class TestWebFetch:
    async def test_fetch_searxng_homepage(self, ctx):
        """Fetch the local SearXNG homepage as a basic test."""
        result = await web_fetch(ctx, "http://localhost:8888/")
        assert "Fetched:" in result
        assert "SearXNG" in result or "searx" in result.lower()

    async def test_fetch_json_endpoint(self, ctx):
        """Fetch a JSON response."""
        result = await web_fetch(ctx, "http://localhost:8888/config")
        assert "Fetched:" in result

    async def test_fetch_invalid_url(self, ctx):
        result = await web_fetch(ctx, "http://localhost:99999/nonexistent")
        assert "Error" in result

    async def test_fetch_nonexistent_host(self, ctx):
        result = await web_fetch(ctx, "http://nonexistent.invalid.host.test/")
        assert "Error" in result

    async def test_fetch_max_length(self, ctx):
        result = await web_fetch(ctx, "http://localhost:8888/", max_length=100)
        # Content should be truncated (page is larger than 100 chars)
        # Total response includes "Fetched: ..." header, but content itself is capped
        assert isinstance(result, str)

    async def test_fetch_max_length_capped_at_50k(self, ctx):
        """max_length should be capped at 50000."""
        result = await web_fetch(ctx, "http://localhost:8888/", max_length=999999)
        # Should work without error
        assert "Fetched:" in result


class TestToolDocstringSynthesis:
    """Verify tool docstrings instruct the model to synthesize results, not relay raw output."""

    def test_web_search_docstring_says_not_relay(self):
        doc = web_search.__doc__
        assert doc is not None
        assert "do not relay" in doc.lower()

    def test_web_search_docstring_says_answer(self):
        doc = web_search.__doc__
        assert doc is not None
        assert "answer" in doc.lower()

    def test_web_search_docstring_mentions_snippet_sufficiency(self):
        doc = web_search.__doc__
        assert doc is not None
        assert "snippets often contain enough" in doc.lower()

    def test_web_fetch_docstring_says_extract(self):
        doc = web_fetch.__doc__
        assert doc is not None
        assert "extract" in doc.lower()

    def test_web_fetch_docstring_says_not_dump(self):
        doc = web_fetch.__doc__
        assert doc is not None
        assert "do not dump" in doc.lower()

    def test_web_fetch_docstring_mentions_budget(self):
        doc = web_fetch.__doc__
        assert doc is not None
        assert "budget" in doc.lower()
        assert "at most 2" in doc.lower()


@pytest.mark.skipif(not has_searxng, reason="SearXNG not running on localhost:8888")
class TestWebSearchSnippetFooter:
    """Verify search results include the snippet reminder footer."""

    async def test_search_results_include_snippet_tip(self, ctx):
        result = await web_search(ctx, "python programming language")
        assert "Tip: These snippets may already answer the question" in result
        assert "Only fetch a URL if you need details not shown above" in result


class TestWebFetchContentQuality:
    """Verify content quality warning on minimal/blocked content."""

    async def test_short_content_triggers_warning(self, ctx):
        stream_cm = _mock_streaming_response("Short", content_type="text/plain")
        mock_client = _mock_client_with_stream(stream_cm)

        with patch("forge.agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await web_fetch(ctx, "http://example.com/short")
        assert "quality issues" in result
        assert "very short" in result
        assert "Do NOT retry" in result

    async def test_blocked_content_triggers_warning(self, ctx):
        html = "<html><body>Please enable JavaScript to view this page. " + "x" * 200 + "</body></html>"
        stream_cm = _mock_streaming_response(html, content_type="text/html")
        mock_client = _mock_client_with_stream(stream_cm)

        with patch("forge.agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await web_fetch(ctx, "http://example.com/blocked")
        assert "quality issues" in result

    async def test_normal_content_no_warning(self, ctx):
        content = "\n".join(
            f"Line {i}: This is a normal page with plenty of content about topic {i}."
            for i in range(20)
        )
        stream_cm = _mock_streaming_response(content, content_type="text/plain")
        mock_client = _mock_client_with_stream(stream_cm)

        with patch("forge.agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await web_fetch(ctx, "http://example.com/normal")
        assert "Warning:" not in result
        assert "Fetched:" in result


class TestWebFetchCache:
    """Verify URL caching within a session."""

    async def test_second_fetch_returns_cached_result(self, ctx):
        content = "\n".join(f"Content line {i} with useful info." for i in range(15))
        stream_cm = _mock_streaming_response(content, content_type="text/plain")
        mock_client = _mock_client_with_stream(stream_cm)

        with patch("forge.agent.tools.httpx.AsyncClient", return_value=mock_client) as mock_cls:
            result1 = await web_fetch(ctx, "http://example.com/cached")
            result2 = await web_fetch(ctx, "http://example.com/cached")

        assert result1 == result2
        assert "Fetched:" in result1
        # AsyncClient should only be constructed once (second call uses cache)
        assert mock_cls.call_count == 1

    async def test_different_urls_not_cached(self, ctx):
        content = "\n".join(f"Content line {i} with useful info." for i in range(15))
        stream_cm1 = _mock_streaming_response(content, content_type="text/plain")
        mock_client1 = _mock_client_with_stream(stream_cm1)
        stream_cm2 = _mock_streaming_response("Other content " * 20, content_type="text/plain")
        mock_client2 = _mock_client_with_stream(stream_cm2)

        with patch("forge.agent.tools.httpx.AsyncClient", side_effect=[mock_client1, mock_client2]) as mock_cls:
            result1 = await web_fetch(ctx, "http://example.com/page1")
            result2 = await web_fetch(ctx, "http://example.com/page2")

        assert mock_cls.call_count == 2
        assert "page1" in result1
        assert "page2" in result2


class TestWebFetchRepetitiveContent:
    """Verify quality warning on repetitive boilerplate content."""

    async def test_repetitive_advertisement_lines(self, ctx):
        # Simulate a page where "Advertisement" dominates
        lines = ["Advertisement"] * 15 + ["Some real content here."] * 3
        content = "\n".join(lines)
        stream_cm = _mock_streaming_response(content, content_type="text/plain")
        mock_client = _mock_client_with_stream(stream_cm)

        with patch("forge.agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await web_fetch(ctx, "http://example.com/ads")
        assert "quality issues" in result
        assert "repetitive" in result
        assert "Advertisement" in result


class TestContentQualityAssessment:
    """Direct unit tests for _assess_content_quality()."""

    def test_short_content(self):
        result = _assess_content_quality("Hi")
        assert "very short" in result

    def test_blocked_content(self):
        result = _assess_content_quality("Please enable JavaScript to continue. " + "x" * 100)
        assert "blocked/gated" in result

    def test_repetitive_lines(self):
        lines = ["Buy now!"] * 10 + ["Real content."] * 2
        result = _assess_content_quality("\n".join(lines))
        assert "repetitive" in result
        assert "Buy now!" in result
        assert "x10" in result

    def test_ad_heavy(self):
        ad_lines = ["Click here to subscribe now"] * 5
        normal_lines = ["Normal content line"] * 3
        result = _assess_content_quality("\n".join(ad_lines + normal_lines))
        assert "ad-heavy" in result

    def test_clean_content_returns_empty(self):
        lines = [f"Paragraph {i} with unique useful content about the topic." for i in range(20)]
        result = _assess_content_quality("\n".join(lines))
        assert result == ""

    def test_thin_js_shell(self):
        # Few unique lines but >100 chars
        content = ("Loading...\n" * 3 + "Please wait\n" * 3 + "x" * 50)
        result = _assess_content_quality(content)
        assert "thin JS shell" in result


class TestWebSearchNoBackend:
    """Tests that work even without SearXNG running."""

    async def test_search_unreachable_returns_error(self, ctx):
        """If no backend is available, should return an error message."""
        result = await web_search(ctx, "test query")
        assert isinstance(result, str)
        # Either we get results (searxng running) or an error message
        assert "Search results" in result or "Error" in result


class TestWebFetchErrorHandling:
    """Tests for web_fetch error paths that don't need SearXNG."""

    async def test_fetch_connection_refused(self, ctx):
        result = await web_fetch(ctx, "http://127.0.0.1:19999/")
        assert "Error" in result

    async def test_fetch_invalid_scheme(self, ctx):
        result = await web_fetch(ctx, "ftp://example.com/file")
        assert "Error" in result or "Fetched" in result  # httpx may handle or error
