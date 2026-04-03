"""Agent tools — file I/O, command execution, code search, and web search."""

from __future__ import annotations

import asyncio
import os
import re
import signal
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from pydantic_ai import ModelRetry, RunContext, Tool

from forge.agent.deps import AgentDeps
from forge.agent.hooks import with_hooks
from forge.agent.utils import head_tail_truncate
from forge.log import get_logger

logger = get_logger(__name__)


# --- Edit diff formatting ---

def _format_edit_diff(old: str, new: str, filename: str, max_lines: int = 50) -> str:
    """Generate a truncated unified diff between old and new file content."""
    import difflib

    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines, fromfile=filename, tofile=filename, lineterm=""))
    if not diff:
        return ""
    if len(diff) > max_lines:
        remaining = len(diff) - max_lines
        diff = diff[:max_lines]
        diff.append(f"... ({remaining} more lines)")
    return "```diff\n" + "\n".join(diff) + "\n```"


# --- Semantic exit code interpretation ---

_SEMANTIC_EXIT_CODES: dict[str, dict[int, str]] = {
    "grep": {1: "No matches found"},
    "rg": {1: "No matches found"},
    "diff": {1: "Files differ"},
    "cmp": {1: "Files differ"},
    "test": {1: "Condition is false"},
}


def _interpret_exit_code(command: str, returncode: int) -> str:
    """Return a human-readable exit code line."""
    if returncode == 0:
        return "Exit code: 0"

    # Extract base command name
    base = command.strip().split()[0] if command.strip() else ""
    base = base.rsplit("/", 1)[-1]  # strip path prefix

    meanings = _SEMANTIC_EXIT_CODES.get(base, {})
    if returncode in meanings:
        return f"Exit code: {returncode} ({meanings[returncode]})"

    if returncode < 0:
        import signal as _sig

        try:
            sig_name = _sig.Signals(-returncode).name
            return f"Exit code: {returncode} (killed by {sig_name})"
        except (ValueError, AttributeError):
            pass

    return f"Exit code: {returncode}"


# --- Silent command classification ---

_SILENT_COMMANDS = {
    "mkdir", "touch", "cp", "mv", "ln", "chmod", "chown",
    "git add", "git checkout", "git switch", "git branch",
    "git stash", "git tag", "git rm", "git mv",
    "cd", "pushd", "popd", "export",
}


def _silent_command_label(command: str) -> str:
    """Return a label for commands that produce no output."""
    cmd = command.strip()
    for silent in _SILENT_COMMANDS:
        if cmd == silent or cmd.startswith(silent + " "):
            return "(completed successfully, no output expected)"
    return "(no output)"


# --- Incremental output reader ---


async def _read_output_incremental(
    proc: asyncio.subprocess.Process,
    *,
    max_bytes: int,
    status_tracker: object | None = None,
    status_interval: float = 2.0,
) -> tuple[bytes, bytes, bool]:
    """Read stdout/stderr incrementally with size guard and optional streaming.

    Returns (stdout_bytes, stderr_bytes, was_killed_for_size).
    """
    from forge.agent.status import Phase

    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    total_bytes = 0
    killed = False
    last_status_update = 0.0

    async def _read_stream(stream: asyncio.StreamReader | None, chunks: list[bytes]) -> None:
        nonlocal total_bytes, killed, last_status_update
        if stream is None:
            return
        while True:
            chunk = await stream.read(8192)
            if not chunk:
                break
            chunks.append(chunk)
            total_bytes += len(chunk)

            # Output size kill guard
            if total_bytes > max_bytes:
                killed = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    proc.kill()
                return

            # Streaming status updates
            if status_tracker and hasattr(status_tracker, "set_phase"):
                now = asyncio.get_event_loop().time()
                if (now - last_status_update) >= status_interval:
                    last_line = chunk.decode("utf-8", errors="replace").rstrip().rsplit("\n", 1)[-1]
                    if last_line.strip():
                        status_tracker.set_phase(Phase.TOOL_EXEC, f"run_command: {last_line[:60]}")
                    last_status_update = now

    # Read both streams concurrently
    await asyncio.gather(
        _read_stream(proc.stdout, stdout_chunks),
        _read_stream(proc.stderr, stderr_chunks),
    )

    return b"".join(stdout_chunks), b"".join(stderr_chunks), killed


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

    # Detect image files — return metadata instead of garbled binary
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        size = path.stat().st_size
        return (
            f"[Image file: {path} ({path.suffix.lower()}, {size:,} bytes)]\n"
            "To view this image, the user should reference it with @path syntax in their prompt."
        )

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
    """Replace text in a file. Tries exact match first, then whitespace-normalized
    and fuzzy matching as fallbacks.

    Args:
        file_path: Path to the file (relative to cwd or absolute).
        old_text: The text to find and replace (must be unique in the file).
        new_text: The replacement text.
    """
    from forge.agent.edit_utils import EditMatchError, find_and_replace

    path = _resolve_path(ctx, file_path)
    if not path.exists():
        raise ModelRetry(f"File not found: {path} — check the path and try again")

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading {path}: {e}"

    try:
        new_content, method, warning = find_and_replace(text, old_text, new_text)
    except EditMatchError as e:
        raise ModelRetry(str(e))

    try:
        path.write_text(new_content, encoding="utf-8")
    except Exception as e:
        return f"Error writing {path}: {e}"

    msg = f"Edited {path}"
    if method != "exact":
        msg += f" (matched via {method})"
    if warning:
        msg += f"\nNote: {warning}"

    # Append unified diff for visibility
    diff_str = _format_edit_diff(text, new_content, str(path))
    if diff_str:
        msg += f"\n\n{diff_str}"

    return msg


async def run_command(
    ctx: RunContext[AgentDeps],
    command: str,
    timeout: float = 0,
) -> str:
    """Execute a shell command in the working directory and return stdout + stderr.

    Args:
        command: The shell command to run.
        timeout: Maximum seconds to wait (0 = use default from config).
    """
    from forge.config import settings as _settings

    if timeout <= 0:
        timeout = _settings.agent.run_command_timeout

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=ctx.deps.cwd,
            start_new_session=True,
        )
    except Exception as e:
        return f"Error running command: {e}"

    # Auto-background: if threshold is set, wait that long then background it
    bg_threshold = _settings.agent.run_command_background_threshold
    if bg_threshold > 0:
        try:
            stdout, stderr, was_killed = await asyncio.wait_for(
                _read_output_incremental(
                    proc,
                    max_bytes=_settings.agent.run_command_max_output_bytes,
                    status_tracker=ctx.deps.status_tracker,
                    status_interval=_settings.agent.run_command_status_interval,
                ),
                timeout=bg_threshold,
            )
        except asyncio.TimeoutError:
            # Background the process instead of killing it
            bg_task = asyncio.create_task(
                _read_output_incremental(
                    proc,
                    max_bytes=_settings.agent.run_command_max_output_bytes,
                )
            )
            ctx.deps._background_procs[proc.pid] = (bg_task, proc)
            return (
                f"Command still running after {bg_threshold:.0f}s — moved to background (PID: {proc.pid}).\n"
                f"Use check_background(pid={proc.pid}) to check status."
            )
    else:
        try:
            stdout, stderr, was_killed = await asyncio.wait_for(
                _read_output_incremental(
                    proc,
                    max_bytes=_settings.agent.run_command_max_output_bytes,
                    status_tracker=ctx.deps.status_tracker,
                    status_interval=_settings.agent.run_command_status_interval,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # Kill entire process group, not just the shell
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                proc.kill()
            await proc.communicate()
            return f"Error: Command timed out after {timeout}s"

    await proc.wait()

    # Decode
    stdout_text = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()

    # Build result
    parts: list[str] = []

    # Semantic exit code
    parts.append(_interpret_exit_code(command, proc.returncode))

    # Truncate with head+tail
    if stdout_text:
        parts.append(
            f"stdout:\n{head_tail_truncate(stdout_text, _settings.agent.run_command_stdout_limit)}"
        )

    if stderr_text:
        parts.append(
            f"stderr:\n{head_tail_truncate(stderr_text, _settings.agent.run_command_stderr_limit)}"
        )

    if was_killed:
        mb = _settings.agent.run_command_max_output_bytes / 1_000_000
        parts.append(
            f"Process killed: output exceeded {mb:.0f} MB limit. "
            "Redirect to file: command > output.txt"
        )
    elif not stdout_text and not stderr_text:
        # Silent command detection
        parts.append(_silent_command_label(command))

    return "\n\n".join(parts)


async def search_code(
    ctx: RunContext[AgentDeps],
    pattern: str,
    path: str = ".",
    glob_filter: str = "",
    file_type: str = "",
    context_lines: int = 0,
) -> str:
    """Search file contents using ripgrep. Returns matching lines with file paths and line numbers.

    Args:
        pattern: Regex pattern to search for.
        path: Directory to search in (relative to cwd or absolute).
        glob_filter: Optional glob to filter files (e.g. "*.py", "*.ts").
        file_type: Optional file type filter (e.g. "py", "ts", "rust").
        context_lines: Lines of context around each match (0-10).
    """
    from forge.config import settings as _settings

    search_path = _resolve_path(ctx, path)
    if not search_path.exists():
        return f"Error: Path not found: {search_path}"

    max_matches = _settings.agent.search_max_matches
    cmd_parts = ["rg", "--line-number", "--no-heading", "--color=never", f"--max-count={max_matches}"]
    if glob_filter:
        cmd_parts.extend(["--glob", glob_filter])
    if file_type:
        cmd_parts.extend(["--type", file_type])
    if context_lines > 0:
        cmd_parts.extend(["-C", str(min(context_lines, 10))])
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
    if len(lines) > max_matches:
        lines = lines[:max_matches]
        lines.append(f"... ({len(output.splitlines())} total matches, showing first {max_matches})")

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

    from forge.config import settings as _settings
    limit = _settings.agent.list_files_limit

    total = len(files)
    if total > limit:
        files = files[:limit]
        files.append(f"... ({total} total files, showing first {limit})")

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

    from forge.config import settings as _settings

    params = {
        "q": query,
        "format": "json",
        "categories": "general,it",
        "language": "en",
        "pageno": 1,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        for base_url in _settings.search.searxng_urls:
            try:
                resp = await client.get(f"{base_url}/search", params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])[:max_results]
                    if not results:
                        return f"No results found for: {query}"

                    return _format_search_results(query, results)
            except (httpx.ConnectError, httpx.TimeoutException):
                continue

        # Fallback to DuckDuckGo
        if _settings.search.ddg_enabled:
            ddg_result = await _ddg_search(query, max_results)
            if ddg_result:
                return ddg_result

        return (
            "Error: No search backend available. "
            "Start a local SearXNG instance (docker run -p 8888:8080 searxng/searxng) "
            "or configure a search endpoint. "
            "Do NOT retry — inform the user that web search is unavailable."
        )


def _format_search_results(query: str, results: list[dict]) -> str:
    """Format search results into a readable string."""
    parts = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("content", "No description")
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
        parts.append(f"{i}. {title}\n   {url}\n   {snippet}\n")
    parts.append(
        "---\n"
        "Tip: These snippets may already answer the question. "
        "Only fetch a URL if you need details not shown above."
    )
    return "\n".join(parts)


def _parse_ddg_html(html: str) -> list[tuple[str, str, str]]:
    """Parse DuckDuckGo HTML results into (title, url, snippet) tuples."""
    results: list[tuple[str, str, str]] = []
    # Match result links
    links = re.findall(
        r'<a[^>]+class="result__a"[^>]+href="([^"]*)"[^>]*>(.*?)</a>',
        html, re.DOTALL,
    )
    snippets = re.findall(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        html, re.DOTALL,
    )
    for i, (raw_url, raw_title) in enumerate(links):
        # DDG wraps URLs in redirects: /l/?uddg=<actual_url>&...
        if "uddg=" in raw_url:
            parsed = parse_qs(urlparse(raw_url).query)
            actual_urls = parsed.get("uddg", [])
            url = actual_urls[0] if actual_urls else raw_url
        else:
            url = raw_url
        title = re.sub(r"<[^>]+>", "", raw_title).strip()
        snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""
        if title and url:
            results.append((title, url, snippet))
    return results


async def _ddg_search(query: str, max_results: int) -> str | None:
    """Search DuckDuckGo using the duckduckgo-search library, with HTML fallback."""
    try:
        from duckduckgo_search import AsyncDDGS

        async with AsyncDDGS() as ddgs:
            raw = await ddgs.atext(query, max_results=max_results)
        if not raw:
            return None
        results = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")}
            for r in raw
        ]
        return _format_search_results(query, results)
    except ImportError:
        pass  # Fall through to HTML scraping
    except Exception:
        logger.debug("DDG library search failed", exc_info=True)

    # Fallback: HTML scraping
    return await _ddg_search_html(query, max_results)


async def _ddg_search_html(query: str, max_results: int) -> str | None:
    """Fallback: scrape DuckDuckGo HTML results."""
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; Forge/0.1)",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            if resp.status_code != 200:
                return None
            parsed = _parse_ddg_html(resp.text)
            if not parsed:
                return None
            results = [
                {"title": t, "url": u, "content": s}
                for t, u, s in parsed[:max_results]
            ]
            return _format_search_results(query, results)
    except Exception:
        logger.debug("DDG HTML search failed", exc_info=True)
        return None


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
        logger.debug("trafilatura failed, using fallback", exc_info=True)

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


async def analyze_impact(
    ctx: RunContext[AgentDeps],
    file_path: str,
) -> str:
    """Analyze what depends on a file. Use before modifying shared code to understand
    downstream impact. Shows symbols defined in the file and which other files import them.

    Args:
        file_path: Path to the file to analyze (relative to cwd or absolute).
    """
    from forge.agent.impact import build_impact_report

    report = await build_impact_report(file_path, ctx.deps.cwd)
    return report.format()


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


async def save_memory(
    ctx: RunContext[AgentDeps],
    category: str,
    subject: str,
    content: str,
) -> str:
    """Save a memory for cross-session recall. Use to remember user preferences,
    project decisions, corrections, or pointers to external resources.

    Args:
        category: One of 'feedback', 'project', 'user', 'reference'.
        subject: Brief title for the memory (e.g. "user prefers terse responses").
        content: Full memory content — be specific so future sessions can use it.
    """
    if ctx.deps.memory_db is None or ctx.deps.memory_project is None:
        return "Memory unavailable — database not connected."

    valid = {"feedback", "project", "user", "reference"}
    if category not in valid:
        return f"Invalid category '{category}'. Must be one of: {', '.join(sorted(valid))}"

    try:
        from forge.agent.memory import save_memory_to_db

        mid = await save_memory_to_db(
            ctx.deps.memory_db, ctx.deps.memory_project, category, subject, content,
        )
        return f"Memory saved (id={mid}, category={category}): {subject}"
    except Exception as e:
        return f"Error saving memory: {e}"


async def recall_memories(
    ctx: RunContext[AgentDeps],
    query: str,
    category: str = "",
    limit: int = 5,
) -> str:
    """Search saved memories by semantic similarity. Use to recall prior context,
    user preferences, decisions, or corrections from past sessions.

    Args:
        query: Natural language query describing what you're looking for.
        category: Optional filter — 'feedback', 'project', 'user', or 'reference'.
        limit: Maximum memories to return (default 5).
    """
    if ctx.deps.memory_db is None or ctx.deps.memory_project is None:
        return "Memory unavailable — database not connected."

    try:
        from forge.agent.memory import recall_from_db

        rows = await recall_from_db(
            ctx.deps.memory_db,
            ctx.deps.memory_project,
            query,
            category=category or None,
            limit=limit,
        )
        if not rows:
            return f"No memories found for: {query}"

        parts = [f"Found {len(rows)} memories:\n"]
        for r in rows:
            parts.append(
                f"- [{r.category}] **{r.subject}** (id={r.id}, score={r.score:.2f})\n"
                f"  {r.content}"
            )
        return "\n".join(parts)
    except Exception as e:
        return f"Error recalling memories: {e}"


async def task_create(
    ctx: RunContext[AgentDeps],
    subject: str,
    description: str,
    active_form: str = "",
) -> str:
    """Create a task to track work progress. Use when work involves 3+ steps.

    Args:
        subject: Brief imperative title (e.g. "Fix auth bug in login flow").
        description: Detailed description of what needs to be done.
        active_form: Present continuous form for status display (e.g. "Fixing auth bug").
    """
    store = ctx.deps.task_store
    if store is None:
        return "Task tracking unavailable."

    task = store.create(subject, description, active_form=active_form or None)
    return f"Created task {task.id}: {task.subject}"


async def task_update(
    ctx: RunContext[AgentDeps],
    task_id: str,
    status: str = "",
    subject: str = "",
    description: str = "",
    add_blocked_by: str = "",
) -> str:
    """Update a task's status or details.

    Args:
        task_id: The task ID (e.g. "t1").
        status: New status — 'pending', 'in_progress', 'completed', or 'deleted'.
        subject: New subject (optional).
        description: New description (optional).
        add_blocked_by: Comma-separated task IDs that block this task (e.g. "t1,t2").
    """
    store = ctx.deps.task_store
    if store is None:
        return "Task tracking unavailable."

    kwargs: dict = {}
    if status:
        from forge.agent.task_store import TaskStatus
        try:
            kwargs["status"] = TaskStatus(status)
        except ValueError:
            return f"Invalid status '{status}'. Use: pending, in_progress, completed, deleted"
    if subject:
        kwargs["subject"] = subject
    if description:
        kwargs["description"] = description
    if add_blocked_by:
        kwargs["add_blocked_by"] = [s.strip() for s in add_blocked_by.split(",")]

    task = store.update(task_id, **kwargs)
    if task is None:
        return f"Task {task_id} not found."
    return f"Updated task {task.id}: status={task.status.value}, subject={task.subject}"


async def task_list(ctx: RunContext[AgentDeps]) -> str:
    """List all tasks with their current status."""
    store = ctx.deps.task_store
    if store is None:
        return "Task tracking unavailable."

    tasks = store.list_all()
    if not tasks:
        return "No tasks."

    lines = ["Tasks:\n"]
    for t in tasks:
        icon = {"pending": "○", "in_progress": "◉", "completed": "✓", "deleted": "✗"}
        marker = icon.get(t.status.value, "?")
        blocked = f" [blocked by {','.join(t.blocked_by)}]" if t.blocked_by else ""
        lines.append(f"  {marker} {t.id}: {t.subject} ({t.status.value}){blocked}")
    return "\n".join(lines)


async def task_get(ctx: RunContext[AgentDeps], task_id: str) -> str:
    """Get full details of a task.

    Args:
        task_id: The task ID (e.g. "t1").
    """
    store = ctx.deps.task_store
    if store is None:
        return "Task tracking unavailable."

    task = store.get(task_id)
    if task is None:
        return f"Task {task_id} not found."

    parts = [
        f"Task {task.id}: {task.subject}",
        f"Status: {task.status.value}",
        f"Description: {task.description}",
    ]
    if task.active_form:
        parts.append(f"Active form: {task.active_form}")
    if task.blocked_by:
        parts.append(f"Blocked by: {', '.join(task.blocked_by)}")
    if task.blocks:
        parts.append(f"Blocks: {', '.join(task.blocks)}")
    return "\n".join(parts)


async def check_background(
    ctx: RunContext[AgentDeps],
    pid: int,
) -> str:
    """Check on a background command started by run_command auto-backgrounding.

    Args:
        pid: The process ID returned when the command was backgrounded.
    """
    entry = ctx.deps._background_procs.get(pid)
    if entry is None:
        return f"No background process with PID {pid} found."

    bg_task, proc = entry

    if bg_task.done():
        # Collect results and clean up
        try:
            stdout, stderr, was_killed = bg_task.result()
        except Exception as e:
            del ctx.deps._background_procs[pid]
            return f"Background process {pid} errored: {e}"

        del ctx.deps._background_procs[pid]
        await proc.wait()

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()

        parts = [f"Background process {pid} finished."]
        parts.append(_interpret_exit_code("", proc.returncode))
        if stdout_text:
            from forge.agent.utils import head_tail_truncate as _htt
            parts.append(f"stdout:\n{_htt(stdout_text, 50000)}")
        if stderr_text:
            from forge.agent.utils import head_tail_truncate as _htt2
            parts.append(f"stderr:\n{_htt2(stderr_text, 20000)}")
        if was_killed:
            parts.append("(output was truncated due to size limit)")
        return "\n\n".join(parts)
    else:
        return f"Background process {pid} is still running."


ALL_TOOLS: list[Tool] = [
    # Read-only tools — safe for parallel execution
    Tool(with_hooks(read_file)),
    Tool(with_hooks(search_code)),
    Tool(with_hooks(list_files)),
    Tool(with_hooks(web_search)),
    Tool(with_hooks(web_fetch)),
    Tool(with_hooks(analyze_impact)),
    Tool(with_hooks(check_background)),
    # Write/execute tools — sequential to prevent races
    Tool(with_hooks(write_file), sequential=True),
    Tool(with_hooks(edit_file), sequential=True),
    Tool(with_hooks(run_command), sequential=True),
]

# Read-only tools for planning mode — explore codebase before planning
READ_ONLY_TOOLS: list[Tool] = [
    Tool(with_hooks(read_file)),
    Tool(with_hooks(search_code)),
    Tool(with_hooks(list_files)),
    Tool(with_hooks(analyze_impact)),
]

# Memory tools — added conditionally when DB is available
MEMORY_TOOLS: list[Tool] = [
    Tool(with_hooks(save_memory), sequential=True),
    Tool(with_hooks(recall_memories)),
]

# Task tools — always available
TASK_TOOLS: list[Tool] = [
    Tool(with_hooks(task_create), sequential=True),
    Tool(with_hooks(task_update), sequential=True),
    Tool(with_hooks(task_list)),
    Tool(with_hooks(task_get)),
]


async def delegate(
    ctx: RunContext[AgentDeps],
    task: str,
    model: str | None = None,
    agent_type: str = "coder",
) -> str:
    """Delegate a contained task to a sub-agent.

    The sub-agent runs in an isolated git worktree with its own tools.
    Use this for self-contained implementation tasks with a clear spec
    (e.g. "add a docstring to all functions in X", "write tests for Y").
    Do NOT delegate tasks requiring cross-file architectural decisions.

    Args:
        ctx: The run context.
        task: Clear description of what the sub-agent should do.
        model: Optional model override (defaults to heavy model via config).
        agent_type: Sub-agent type — 'coder' (read+write+run), 'research' (read-only), or 'reviewer' (read+run).

    Returns:
        Sub-agent's output summary, including branch name if changes were made.
    """
    from forge.agent.subagent import run_subagent_and_merge

    result = await run_subagent_and_merge(
        task=task,
        cwd=ctx.deps.cwd,
        model=model,
        parent_hooks=ctx.deps.hook_registry,
        mcp_servers=ctx.deps.mcp_servers,
        agent_type=agent_type,
    )
    if not result.success:
        raise ModelRetry(f"Sub-agent failed: {result.output}")
    return result.output


async def delegate_parallel(
    ctx: RunContext[AgentDeps],
    tasks: list[str],
    model: str | None = None,
) -> str:
    """Delegate multiple independent tasks to sub-agents running in parallel.

    Each task runs in its own isolated git worktree. Use when tasks are
    independent and don't modify the same files. Max 4 concurrent.

    Args:
        ctx: The run context.
        tasks: List of task descriptions (2-4 tasks).
        model: Optional model override.

    Returns:
        Combined summary of all sub-agent results.
    """
    if len(tasks) < 2:
        raise ModelRetry("Use 'delegate' for single tasks. delegate_parallel requires 2+ tasks.")
    if len(tasks) > 6:
        raise ModelRetry("Max 6 parallel tasks. Split into batches if needed.")

    from forge.agent.subagent import run_subagents_parallel

    results = await run_subagents_parallel(
        tasks=tasks,
        cwd=ctx.deps.cwd,
        model=model,
        parent_hooks=ctx.deps.hook_registry,
        mcp_servers=ctx.deps.mcp_servers,
    )

    parts = []
    any_failed = False
    for i, r in enumerate(results):
        status = "OK" if r.success else "FAILED"
        parts.append(f"### Task {i + 1} [{status}]\n{r.output}")
        if not r.success:
            any_failed = True

    summary = "\n\n".join(parts)
    if any_failed:
        summary += "\n\n**Note:** Some tasks failed. Review above and retry individually if needed."
    return summary


DELEGATE_TOOLS: list[Tool] = [
    Tool(with_hooks(delegate), sequential=True),
    Tool(with_hooks(delegate_parallel), sequential=True),
]
