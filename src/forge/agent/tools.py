"""Agent tools — file I/O, command execution, code search, and web search."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
from pydantic_ai import ModelRetry, RunContext, Tool

from forge.agent.deps import AgentDeps
from forge.agent.permissions import check_permission


def _resolve_path(ctx: RunContext[AgentDeps], file_path: str) -> Path:
    """Resolve a file path relative to the agent's working directory."""
    p = Path(file_path)
    if not p.is_absolute():
        p = ctx.deps.cwd / p
    return p.resolve()


async def read_file(
    ctx: RunContext[AgentDeps],
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read a file and return its contents with line numbers.

    Args:
        file_path: Path to the file (relative to cwd or absolute).
        offset: Line number to start reading from (0-based).
        limit: Maximum number of lines to return.
    """
    path = _resolve_path(ctx, file_path)
    if not path.exists():
        raise ModelRetry(f"File not found: {path} — check the path and try again")
    if not path.is_file():
        raise ModelRetry(f"Not a file: {path} — this is a directory, use list_files instead")

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading {path}: {e}"

    lines = text.splitlines()
    total = len(lines)
    selected = lines[offset : offset + limit]

    numbered = []
    for i, line in enumerate(selected, start=offset + 1):
        numbered.append(f"{i:>6}\t{line}")

    header = f"File: {path} ({total} lines)"
    if offset > 0 or len(selected) < total:
        header += f" [showing lines {offset + 1}-{offset + len(selected)}]"

    return f"{header}\n{'─' * 40}\n" + "\n".join(numbered)


async def write_file(
    ctx: RunContext[AgentDeps],
    file_path: str,
    content: str,
) -> str:
    """Create or overwrite a file with the given content.

    Args:
        file_path: Path to the file (relative to cwd or absolute).
        content: The full content to write to the file.
    """
    if not await check_permission(
        ctx.deps.console, ctx.deps.permission, "write_file",
        {"file_path": file_path, "content": content},
    ):
        return "Permission denied by user."

    path = _resolve_path(ctx, file_path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception as e:
        return f"Error writing {path}: {e}"

    lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return f"Wrote {lines} lines to {path}"


async def edit_file(
    ctx: RunContext[AgentDeps],
    file_path: str,
    old_text: str,
    new_text: str,
) -> str:
    """Replace an exact string in a file. The old_text must appear exactly once.

    Args:
        file_path: Path to the file (relative to cwd or absolute).
        old_text: The exact text to find and replace (must be unique in the file).
        new_text: The replacement text.
    """
    if not await check_permission(
        ctx.deps.console, ctx.deps.permission, "edit_file",
        {"file_path": file_path, "old_text": old_text, "new_text": new_text},
    ):
        return "Permission denied by user."

    path = _resolve_path(ctx, file_path)
    if not path.exists():
        raise ModelRetry(f"File not found: {path} — check the path and try again")

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading {path}: {e}"

    count = text.count(old_text)
    if count == 0:
        raise ModelRetry(
            f"old_text not found in {path} — read the file first to get the exact text"
        )
    if count > 1:
        raise ModelRetry(
            f"old_text appears {count} times in {path} — provide more surrounding "
            "context to make the match unique"
        )

    new_content = text.replace(old_text, new_text, 1)
    try:
        path.write_text(new_content, encoding="utf-8")
    except Exception as e:
        return f"Error writing {path}: {e}"

    return f"Edited {path} — replaced 1 occurrence"


async def run_command(
    ctx: RunContext[AgentDeps],
    command: str,
    timeout: float = 30.0,
) -> str:
    """Execute a shell command in the working directory and return stdout + stderr.

    Args:
        command: The shell command to run.
        timeout: Maximum seconds to wait (default 30).
    """
    if not await check_permission(
        ctx.deps.console, ctx.deps.permission, "run_command",
        {"command": command},
    ):
        return "Permission denied by user."

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=ctx.deps.cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error running command: {e}"

    parts: list[str] = []
    parts.append(f"Exit code: {proc.returncode}")

    stdout_text = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()

    if stdout_text:
        # Truncate very long output
        if len(stdout_text) > 10_000:
            stdout_text = stdout_text[:10_000] + "\n... (truncated)"
        parts.append(f"stdout:\n{stdout_text}")

    if stderr_text:
        if len(stderr_text) > 5_000:
            stderr_text = stderr_text[:5_000] + "\n... (truncated)"
        parts.append(f"stderr:\n{stderr_text}")

    if not stdout_text and not stderr_text:
        parts.append("(no output)")

    return "\n\n".join(parts)


async def search_code(
    ctx: RunContext[AgentDeps],
    pattern: str,
    path: str = ".",
    glob_filter: str = "",
) -> str:
    """Search file contents using ripgrep. Returns matching lines with file paths and line numbers.

    Args:
        pattern: Regex pattern to search for.
        path: Directory to search in (relative to cwd or absolute).
        glob_filter: Optional glob to filter files (e.g. "*.py", "*.ts").
    """
    search_path = _resolve_path(ctx, path)
    if not search_path.exists():
        return f"Error: Path not found: {search_path}"

    cmd_parts = ["rg", "--line-number", "--no-heading", "--color=never", "--max-count=50"]
    if glob_filter:
        cmd_parts.extend(["--glob", glob_filter])
    cmd_parts.extend(["--", pattern, str(search_path)])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=ctx.deps.cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)
    except FileNotFoundError:
        return "Error: ripgrep (rg) not found — install it with your package manager"
    except asyncio.TimeoutError:
        return "Error: Search timed out after 15s"
    except Exception as e:
        return f"Error: {e}"

    output = stdout.decode("utf-8", errors="replace").strip()
    if not output:
        if proc.returncode == 1:
            return "No matches found."
        err = stderr.decode("utf-8", errors="replace").strip()
        return f"Error: {err}" if err else "No matches found."

    lines = output.splitlines()
    if len(lines) > 50:
        lines = lines[:50]
        lines.append(f"... ({len(output.splitlines())} total matches, showing first 50)")

    return "\n".join(lines)


async def list_files(
    ctx: RunContext[AgentDeps],
    pattern: str = "**/*",
    path: str = ".",
) -> str:
    """List files matching a glob pattern.

    Args:
        pattern: Glob pattern (default: "**/*" for all files).
        path: Directory to search in (relative to cwd or absolute).
    """
    search_path = _resolve_path(ctx, path)
    if not search_path.exists():
        return f"Error: Path not found: {search_path}"

    try:
        matches = sorted(search_path.glob(pattern))
    except Exception as e:
        return f"Error: {e}"

    # Filter to files only, skip hidden dirs and common noise
    skip_dirs = {".git", "__pycache__", "node_modules", ".venv", ".mypy_cache", ".ruff_cache"}
    files = []
    for m in matches:
        if m.is_file() and not any(part in skip_dirs for part in m.parts):
            files.append(str(m.relative_to(search_path)))

    if not files:
        return f"No files matching '{pattern}' in {search_path}"

    total = len(files)
    if total > 200:
        files = files[:200]
        files.append(f"... ({total} total files, showing first 200)")

    return "\n".join(files)


async def web_search(
    ctx: RunContext[AgentDeps],
    query: str,
    max_results: int = 5,
) -> str:
    """Search the web for information. Use for weather, news, documentation lookups, API references,
    error messages, or any question requiring up-to-date information.

    Search snippets often contain enough information to answer directly — only use web_fetch
    if the snippets lack the specific detail you need.

    The results are raw search data — use them to formulate a direct answer to the user's question.
    Do NOT relay the raw results to the user; instead, read the results and answer their question.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (default 5, max 10).
    """
    max_results = min(max_results, 10)

    # SearXNG — local, no API key needed
    # Falls back to common public instances if local isn't running
    searxng_urls = [
        "http://localhost:8888",  # common local SearXNG
        "http://localhost:8080",  # alternative local port
    ]

    params = {
        "q": query,
        "format": "json",
        "categories": "general,it",
        "language": "en",
        "pageno": 1,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        for base_url in searxng_urls:
            try:
                resp = await client.get(f"{base_url}/search", params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])[:max_results]
                    if not results:
                        return f"No results found for: {query}"

                    parts = [f"Search results for: {query}\n"]
                    for i, r in enumerate(results, 1):
                        title = r.get("title", "Untitled")
                        url = r.get("url", "")
                        snippet = r.get("content", "No description")
                        # Truncate long snippets
                        if len(snippet) > 500:
                            snippet = snippet[:500] + "..."
                        parts.append(f"{i}. {title}\n   {url}\n   {snippet}\n")

                    parts.append(
                        "---\n"
                        "Tip: These snippets may already answer the question. "
                        "Only fetch a URL if you need details not shown above."
                    )

                    return "\n".join(parts)
            except (httpx.ConnectError, httpx.TimeoutException):
                continue

        return (
            "Error: No search backend available. "
            "Start a local SearXNG instance (docker run -p 8888:8080 searxng/searxng) "
            "or configure a search endpoint. "
            "Do NOT retry — inform the user that web search is unavailable."
        )


def _assess_content_quality(text: str) -> str:
    """Check extracted text for quality issues and return a warning string (empty if OK)."""
    stripped = text.strip()
    stripped_lower = stripped.lower()
    warnings: list[str] = []

    # Short content
    if len(stripped) < 100:
        warnings.append("very short")

    # Blocked/gated content
    blockers = (
        "enable javascript", "access denied", "captcha", "please verify",
        "just a moment", "checking your browser", "ray id",
    )
    if any(m in stripped_lower for m in blockers):
        warnings.append("blocked/gated")

    # Thin JS shell
    unique_lines = set(line.strip() for line in stripped.splitlines() if line.strip())
    if len(stripped) > 100 and len(unique_lines) < 8:
        warnings.append("thin JS shell")

    # Repetitive content
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) >= 5:
        from collections import Counter

        counts = Counter(lines)
        most_common_line, most_common_count = counts.most_common(1)[0]
        if most_common_count >= 3 and most_common_count / len(lines) > 0.3:
            warnings.append(
                f"repetitive ('{most_common_line[:40]}' x{most_common_count})"
            )

    # Ad-heavy content
    ad_markers = ("advertisement", "sponsored", "subscribe now", "cookie", "sign up", "log in")
    if lines:
        ad_lines = sum(1 for l in lines if any(m in l.lower() for m in ad_markers))
        if ad_lines / len(lines) > 0.25:
            warnings.append(f"ad-heavy ({ad_lines}/{len(lines)} lines)")

    if not warnings:
        return ""
    return (
        f"Warning: This page has quality issues ({', '.join(warnings)}). "
        "Do NOT retry — use the information you already have.\n\n"
    )


async def web_fetch(
    ctx: RunContext[AgentDeps],
    url: str,
    max_length: int = 10000,
) -> str:
    """Fetch a web page and return its text content. Only fetch when search snippets are
    insufficient. Budget: at most 2 fetches per question.

    The fetched content is raw data for you to process. Extract the relevant information
    and present a clear answer to the user — do not dump raw page content.

    Args:
        url: The URL to fetch.
        max_length: Maximum characters to return (default 10000).
    """
    max_length = min(max_length, 50000)

    # Check cache
    if url in ctx.deps.url_cache:
        return ctx.deps.url_cache[url]

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; Forge/0.1; +https://github.com)",
        "Accept": "text/html,text/plain,application/json",
    }
    MAX_DOWNLOAD = 1_000_000  # 1MB cap

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, max_redirects=5) as client:
        try:
            async with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code >= 400:
                    return (
                        f"Error: HTTP {resp.status_code} fetching {url}. "
                        "Do NOT retry this URL — report the error to the user and move on."
                    )

                chunks: list[bytes] = []
                size = 0
                async for chunk in resp.aiter_bytes(4096):
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > MAX_DOWNLOAD:
                        break
                raw_bytes = b"".join(chunks)
        except httpx.TimeoutException:
            return f"Error: Request timed out fetching {url}. Do NOT retry — inform the user."
        except httpx.ConnectError as e:
            return f"Error: Could not connect to {url}: {e}. Do NOT retry — inform the user."
        except Exception as e:
            return f"Error fetching {url}: {e}. Do NOT retry — inform the user."

        content_type = resp.headers.get("content-type", "")
        text = raw_bytes.decode("utf-8", errors="replace")

        if "json" in content_type:
            pass  # text as-is
        elif "html" in content_type:
            text = _strip_html(text, url=url)

        if len(text) > max_length:
            text = text[:max_length] + f"\n... (truncated, {len(text)} total chars)"

        quality_warning = _assess_content_quality(text)
        result = f"{quality_warning}Fetched: {url}\n{'─' * 40}\n{text}"

        # Cache result
        ctx.deps.url_cache[url] = result
        return result


def _strip_html(html: str, url: str | None = None) -> str:
    """Extract main content from HTML, stripping boilerplate navigation/menus.

    Uses trafilatura for intelligent content extraction, with a regex fallback.
    """
    try:
        import trafilatura

        text = trafilatura.extract(
            html,
            url=url,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
            deduplicate=True,
        )
        if text and len(text) > 50:
            return text
    except Exception:
        pass

    # Fallback: aggressive regex stripping
    return _strip_html_fallback(html)


def _strip_html_fallback(html: str) -> str:
    """Regex-based HTML-to-text fallback."""
    import re

    # Remove script, style, nav, header, footer, aside blocks entirely
    for tag in ("script", "style", "nav", "header", "footer", "aside", "noscript"):
        html = re.sub(rf"<{tag}[^>]*>[\s\S]*?</{tag}>", "", html, flags=re.IGNORECASE)
    # Remove hidden elements
    html = re.sub(r'<[^>]+(?:hidden|display:\s*none)[^>]*>[\s\S]*?</[^>]+>', "", html)
    # Remove all HTML tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Decode common entities
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                         ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " "),
                         ("&deg;", "°")]:
        text = text.replace(entity, char)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def rag_search(
    ctx: RunContext[AgentDeps],
    query: str,
    limit: int = 5,
) -> str:
    """Search the indexed codebase semantically. Returns relevant code chunks ranked by similarity.

    Use this for conceptual queries like "how does routing work" or "where is authentication handled".
    Use search_code for exact text/regex matches instead.

    Args:
        query: Natural language query describing what you're looking for.
        limit: Maximum number of chunks to return (default 5).
    """
    if ctx.deps.rag_db is None or ctx.deps.rag_project is None:
        return "RAG search unavailable — project is not indexed. Use search_code for text search."

    try:
        from forge.rag.retriever import format_context, retrieve

        chunks = await retrieve(
            query,
            ctx.deps.rag_project,
            ctx.deps.rag_db,  # type: ignore[arg-type]
            limit=limit,
        )
        if not chunks:
            return f"No relevant code found for: {query}"

        return format_context(chunks)
    except Exception as e:
        return f"RAG search error: {e}. Use search_code as fallback."


ALL_TOOLS: list[Tool | object] = [
    # Read-only tools — safe for parallel execution
    read_file,
    search_code,
    list_files,
    web_search,
    web_fetch,
    # Write/execute tools — sequential to prevent races
    Tool(write_file, sequential=True),
    Tool(edit_file, sequential=True),
    Tool(run_command, sequential=True),
]
