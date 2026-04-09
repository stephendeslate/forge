"""Forge MCP server — expose Forge tools to external clients.

Provides file I/O, code search, command execution, plus Forge-specific
capabilities: local model inference, RAG search, and cross-session memory.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

from fastmcp import FastMCP

from forge.config import settings
from forge.log import get_logger

logger = get_logger(__name__)

mcp = FastMCP("Forge")

# Working directory — set at startup, tools operate relative to this
_cwd: Path = Path.cwd()

# Lazy-initialized database connection
_db = None
_db_lock = asyncio.Lock()


def _resolve(path: str) -> Path:
    """Resolve a path relative to the working directory."""
    p = Path(path)
    if not p.is_absolute():
        p = _cwd / p
    return p.resolve()


async def _get_db():
    """Lazy-init database connection pool."""
    global _db
    if _db is not None:
        return _db

    async with _db_lock:
        if _db is not None:
            return _db
        try:
            from forge.storage.database import Database
            db = Database(settings.db.dsn)
            await db.connect()
            _db = db
            return _db
        except Exception as e:
            logger.warning("Database unavailable for MCP server: %s", e)
            return None


def _project_name() -> str:
    """Derive project name from working directory."""
    return _cwd.name


# ---------------------------------------------------------------------------
# File I/O tools
# ---------------------------------------------------------------------------


@mcp.tool()
def read_file(
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read a file and return its contents with line numbers.

    Args:
        file_path: Path to the file (relative to working directory or absolute).
        offset: Line offset to start reading from (0-based).
        limit: Maximum number of lines to return.
    """
    from forge.agent.sandbox import check_path_boundary
    boundary_err = check_path_boundary(file_path, _cwd)
    if boundary_err:
        return f"Error: {boundary_err}"
    resolved = _resolve(file_path)
    if not resolved.exists():
        return f"Error: File not found: {file_path}"
    if resolved.is_dir():
        return f"Error: {file_path} is a directory, not a file."

    # Detect image files — return metadata instead of garbled binary
    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
    if resolved.suffix.lower() in _IMAGE_EXTS:
        size = resolved.stat().st_size
        return f"[Image file: {resolved} ({resolved.suffix.lower()}, {size:,} bytes)]"

    try:
        lines = resolved.read_text().splitlines()
    except Exception as e:
        return f"Error reading {file_path}: {e}"

    total = len(lines)
    selected = lines[offset : offset + limit]
    numbered = [f"{i + offset + 1:>6}\t{line}" for i, line in enumerate(selected)]
    header = f"File: {file_path} ({total} lines)"
    if offset > 0 or len(selected) < total:
        header += f" — showing lines {offset + 1}-{offset + len(selected)}"
    return header + "\n" + "\n".join(numbered)


@mcp.tool()
def write_file(file_path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed.

    Args:
        file_path: Path to the file.
        content: The content to write.
    """
    from forge.agent.sandbox import check_path_boundary
    boundary_err = check_path_boundary(file_path, _cwd)
    if boundary_err:
        return f"Error: {boundary_err}"
    resolved = _resolve(file_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content)
    lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return f"Wrote {lines} lines to {file_path}"


@mcp.tool()
def edit_file(file_path: str, old_text: str, new_text: str) -> str:
    """Replace text in a file. Tries exact match first, then whitespace-normalized
    and fuzzy matching as fallbacks.

    Args:
        file_path: Path to the file.
        old_text: The text to find and replace.
        new_text: The replacement text.
    """
    from forge.agent.edit_utils import EditMatchError, find_and_replace
    from forge.agent.sandbox import check_path_boundary
    boundary_err = check_path_boundary(file_path, _cwd)
    if boundary_err:
        return f"Error: {boundary_err}"

    resolved = _resolve(file_path)
    if not resolved.exists():
        return f"Error: File not found: {file_path}"

    content = resolved.read_text()
    try:
        new_content, method, warning = find_and_replace(content, old_text, new_text)
    except EditMatchError as e:
        return f"Error: {e}"

    resolved.write_text(new_content)
    msg = f"Edited {file_path}"
    if method != "exact":
        msg += f" (matched via {method})"
    if warning:
        msg += f"\nNote: {warning}"

    # Append unified diff for visibility
    import difflib
    old_lines = content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines, fromfile=file_path, tofile=file_path, lineterm=""))
    if diff:
        msg += "\n\n" + "\n".join(diff[:50])
        if len(diff) > 50:
            msg += f"\n... ({len(diff)} diff lines total)"
    return msg


@mcp.tool()
def search_code(
    pattern: str,
    path: str = ".",
    glob_filter: str | None = None,
) -> str:
    """Search for a regex pattern in files using ripgrep.

    Args:
        pattern: Regex pattern to search for.
        path: Directory to search in (relative to working directory).
        glob_filter: Optional glob to filter files (e.g. "*.py").
    """
    resolved = _resolve(path)
    from forge.agent.sandbox import check_path_boundary
    boundary_err = check_path_boundary(str(resolved), _cwd)
    if boundary_err:
        return f"Error: {boundary_err}"
    from forge.config import settings as _settings
    max_matches = _settings.agent.search_max_matches
    cmd = ["rg", "--no-heading", "-n", "--color=never", f"--max-count={max_matches}", pattern, str(resolved)]
    if glob_filter:
        cmd.extend(["--glob", glob_filter])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, cwd=str(_cwd)
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            lines = output.split("\n")
            if len(lines) > max_matches:
                output = "\n".join(lines[:max_matches]) + f"\n... ({len(lines)} total matches)"
            return output or "No matches found."
        if result.returncode == 1:
            return "No matches found."
        return f"Error: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "Error: Search timed out."
    except FileNotFoundError:
        return "Error: ripgrep (rg) not installed."


@mcp.tool()
def list_files(path: str = ".", pattern: str = "**/*") -> str:
    """List files matching a glob pattern.

    Args:
        path: Directory to list from.
        pattern: Glob pattern (default: all files).
    """
    resolved = _resolve(path)
    from forge.agent.sandbox import check_path_boundary
    boundary_err = check_path_boundary(str(resolved), _cwd)
    if boundary_err:
        return f"Error: {boundary_err}"
    if not resolved.is_dir():
        return f"Error: Not a directory: {path}"

    skip = {".git", "__pycache__", "node_modules", ".venv", ".forge", ".mypy_cache", ".ruff_cache"}
    files: list[str] = []
    for p in sorted(resolved.glob(pattern)):
        if any(part in skip for part in p.parts):
            continue
        if p.is_file():
            try:
                rel = p.relative_to(resolved)
            except ValueError:
                rel = p
            files.append(str(rel))
            if len(files) >= 500:
                break

    if not files:
        return f"No files matching '{pattern}' in {path}"
    result = f"{len(files)} files"
    if len(files) >= 500:
        result += " (truncated at 500)"
    return result + ":\n" + "\n".join(files)


@mcp.tool()
def run_command(command: str, timeout: float = 30.0) -> str:
    """Run a shell command in the working directory.

    Args:
        command: The shell command to execute.
        timeout: Maximum execution time in seconds.
    """
    from forge.agent.sandbox import check_command_blocklist
    block_err = check_command_blocklist(command)
    if block_err:
        return f"Error: {block_err}"
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(_cwd),
            start_new_session=True,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        output = output.strip()
        max_output = 50000
        if len(output) > max_output:
            output = output[:max_output] + "\n... (truncated)"
        return f"{output}\n\nExit code: {result.returncode}"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Forge-specific tools — local inference, RAG, memory
# ---------------------------------------------------------------------------


@mcp.tool()
async def forge_ask(
    prompt: str,
    model: str = "auto",
    system_prompt: str = "",
) -> str:
    """Send a prompt to a local model and get a response. Free and fast.

    Use this for tasks that don't need cloud intelligence — code generation,
    explanations, reformatting, summarization, etc.

    Args:
        prompt: The question or task.
        model: "heavy", "fast", "npu", or "auto" (routes by complexity).
        system_prompt: Optional system prompt override.
    """
    from forge.core.router import Route
    from forge.models.npu import get_npu_backend
    from forge.models.ollama import get_fast_backend, get_heavy_backend
    from forge.prompts.system import CHAT_SYSTEM

    system = system_prompt or CHAT_SYSTEM

    route_map = {
        "heavy": Route.HEAVY,
        "fast": Route.FAST,
        "npu": Route.NPU,
    }
    force_route = route_map.get(model)

    # Build router and route
    from forge.core.router import ModelRouter
    router = ModelRouter(heavy=get_heavy_backend(), fast=get_fast_backend(), npu=get_npu_backend())
    route, backend = router.route(prompt, force=force_route)

    try:
        result = await backend.generate(prompt, system=system)
        return result
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
async def forge_rag_search(
    query: str,
    limit: int = 5,
) -> str:
    """Search the indexed codebase using semantic similarity.

    Returns relevant code chunks ranked by hybrid vector + BM25 search.
    Index the project first with `forge index .`

    Args:
        query: Natural language search query.
        limit: Max results to return (default: 5).
    """
    db = await _get_db()
    if db is None:
        return "Error: Database unavailable. RAG search requires PostgreSQL."

    project = _project_name()

    try:
        from forge.rag.retriever import retrieve
        chunks = await retrieve(query, project, db, limit=limit)
    except Exception as e:
        return f"Error: RAG search failed: {e}"

    if not chunks:
        return f"No results found for '{query}'. Is the project indexed? Run: forge index ."

    parts = []
    for chunk in chunks:
        score = f"{chunk.score:.2f}" if hasattr(chunk, "score") and chunk.score else ""
        header = f"# {chunk.file_path}"
        if hasattr(chunk, "name") and chunk.name:
            header += f" — {chunk.name}"
        if hasattr(chunk, "start_line") and chunk.start_line:
            header += f" (lines {chunk.start_line}-{chunk.end_line})"
        if score:
            header += f" [score: {score}]"
        parts.append(f"{header}\n```\n{chunk.content}\n```")

    return "\n\n".join(parts)


@mcp.tool()
async def forge_memory_recall(
    query: str,
    category: str = "",
    limit: int = 5,
) -> str:
    """Recall memories from Forge's cross-session memory system.

    Searches past observations, feedback, and project knowledge using
    semantic similarity.

    Args:
        query: What to search for.
        category: Optional filter — one of: feedback, project, user, reference.
        limit: Max memories to return.
    """
    db = await _get_db()
    if db is None:
        return "Error: Database unavailable. Memory requires PostgreSQL."

    project = _project_name()

    try:
        from forge.agent.memory import recall_from_db
        rows = await recall_from_db(
            db, project, query,
            category=category or None,
            limit=limit,
        )
    except Exception as e:
        return f"Error: Memory recall failed: {e}"

    if not rows:
        return f"No memories found for '{query}'."

    parts = []
    for r in rows:
        score = f" (similarity: {r.score:.2f})" if r.score else ""
        parts.append(f"[{r.category}] **{r.subject}**{score}\n{r.content}")

    return "\n\n---\n\n".join(parts)


@mcp.tool()
async def forge_memory_save(
    category: str,
    subject: str,
    content: str,
) -> str:
    """Save a memory to Forge's cross-session memory system.

    Memories persist across sessions and are recalled by semantic similarity.

    Args:
        category: One of: feedback, project, user, reference.
        subject: Brief title for the memory.
        content: The memory content.
    """
    valid_categories = {"feedback", "project", "user", "reference"}
    if category not in valid_categories:
        return f"Error: category must be one of: {', '.join(sorted(valid_categories))}"

    db = await _get_db()
    if db is None:
        return "Error: Database unavailable. Memory requires PostgreSQL."

    project = _project_name()

    try:
        from forge.agent.memory import save_memory_to_db
        memory_id = await save_memory_to_db(db, project, category, subject, content)
        return f"Memory saved (id: {memory_id}, category: {category}, subject: {subject})"
    except Exception as e:
        return f"Error: Memory save failed: {e}"


# ---------------------------------------------------------------------------
# Resource
# ---------------------------------------------------------------------------


@mcp.resource("forge://status")
def forge_status() -> str:
    """Current Forge configuration and working directory."""
    return (
        f"Working directory: {_cwd}\n"
        f"Heavy model: {settings.ollama.heavy_model}\n"
        f"Fast model: {settings.ollama.fast_model}\n"
        f"Embed model: {settings.ollama.embed_model}\n"
        f"Ollama URL: {settings.ollama.base_url}\n"
        f"Mode: {settings.agent.mode}\n"
    )


def run(cwd: str | None = None) -> None:
    """Entry point for the MCP server."""
    global _cwd
    if cwd:
        _cwd = Path(cwd).resolve()
    mcp.run()
